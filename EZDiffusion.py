import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import os
from glob import glob

# 简化中文字体配置（仅保留系统最常见的SimHei）
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class EZDiffusionAnalyzer:
    """EZ-Diffusion模型分析器，用于计算决策阈值等参数并检测阈值塌陷"""
    
    def __init__(self, data_path=None):
        self.data = None
        self.params = None
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """加载实验数据"""
        if os.path.isdir(data_path):
            files = glob(os.path.join(data_path, "*.csv"))
            if not files:
                raise ValueError("指定目录中没有找到CSV文件")
            self.data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        else:
            self.data = pd.read_csv(data_path)
        
        self._preprocess_data()
        print(f"成功加载数据，共{len(self.data)}条有效试次")
        return self
    
    def _preprocess_data(self):
        """数据预处理：过滤无效试次，计算必要指标"""
        self.data = self.data[self.data['rt'] > 0].copy()
        
        # 反应时单位转换（毫秒→秒）
        if self.data['rt'].mean() > 100:
            self.data['rt'] = self.data['rt'] / 1000
        
        # 定义正确反应（跨期决策：选择延迟选项为正确）
        self.data['correct'] = self.data['chose_delayed'].astype(bool)
        
        # 强制所有条件列转为字符串，避免拆分报错
        self.data['subj'] = self.data['participant'].astype(str)
        self.data['block_sign'] = self.data['block_sign'].astype(str)
        self.data['block_pressure'] = self.data['block_pressure'].astype(str)
        self.data['condition'] = self.data['block_sign'] + "_" + self.data['block_pressure']
        
        # 试次序号（按被试+条件分组）
        self.data['trial_num'] = self.data.groupby(['subj', 'condition']).cumcount() + 1
    
    def calculate_ez_parameters(self, groupby_cols=['subj', 'condition']):
        """计算EZ-Diffusion模型参数（边界分离a、漂移率v、非决策时间ter）"""
        # 分组计算统计量（兼容低版本pandas，无include_groups=False）
        grouped = self.data.groupby(groupby_cols).apply(self._ez_stats).reset_index()
        
        # 计算EZ参数
        grouped = grouped.apply(self._ez_calculate_params, axis=1)
        
        # 拆分condition列为block_sign/block_pressure（逐行处理，避免类型错误）
        if 'condition' in grouped.columns:
            split_cols = grouped['condition'].str.split('_', expand=True).astype(str)
            if split_cols.shape[1] >= 2:
                grouped['block_sign'] = split_cols[0]
                # 拼接压力条件（过滤空值）
                def join_pressure(row):
                    parts = [p for p in row[1:] if p != 'nan']
                    return '_'.join(parts) if parts else 'unknown'
                grouped['block_pressure'] = split_cols.apply(join_pressure, axis=1)
        
        # 过滤全NaN参数行
        before_filter = len(grouped)
        self.params = grouped.dropna(subset=['a', 'v', 'ter'], how='all')
        after_filter = len(self.params)
        if before_filter - after_filter > 0:
            print(f"\n提示：{before_filter - after_filter}组数据因统计量不足无法计算参数，已过滤")
        
        return self.params
    
    def _ez_stats(self, df):
        """计算EZ模型所需统计量（正确率、反应时均值/方差）"""
        n = len(df)
        p_correct = df['correct'].mean()
        
        # 确保正确率≥0.5（反转正确/错误定义）
        if p_correct < 0.5:
            df = df.copy()
            df['correct'] = ~df['correct']
            p_correct = df['correct'].mean()
        
        # 限制极端正确率（避免计算错误）
        p_correct = np.clip(p_correct, 0.51, 0.99)
        
        # 正确/错误反应时的均值和方差
        rt_correct = df[df['correct']]['rt']
        rt_error = df[~df['correct']]['rt']
        
        return pd.Series({
            'n': n,
            'p_correct': p_correct,
            'mean_correct': rt_correct.mean() if len(rt_correct) > 0 else np.nan,
            'mean_error': rt_error.mean() if len(rt_error) > 0 else np.nan,
            'var_correct': rt_correct.var() if len(rt_correct) > 1 else np.nan
        })
    
    def _ez_calculate_params(self, row):
        """根据EZ公式计算扩散模型参数"""
        if any(np.isnan([row['mean_correct'], row['mean_error'], row['var_correct']])):
            row['a'] = row['v'] = row['ter'] = np.nan
            return row
        
        # EZ-Diffusion核心公式（Wagenmakers et al., 2007）
        p, m, me, var = row['p_correct'], row['mean_correct'], row['mean_error'], row['var_correct']
        a = (norm.ppf(p) * np.sqrt(2 * var)) / np.sqrt(norm.ppf(p) **2 + norm.ppf(1 - p)** 2)
        ter = m - (a * norm.ppf(p)) ** 2 / (2 * (p * norm.ppf(p) - (1 - p) * norm.ppf(1 - p)))
        v = (p * norm.ppf(p) - (1 - p) * norm.ppf(1 - p)) / (a * np.sqrt(var))
        
        row['a'] = a  # 决策阈值（边界分离）
        row['v'] = v  # 漂移率（信息积累速度）
        row['ter'] = ter  # 非决策时间（感知+运动反应）
        return row
    
    def analyze_threshold_collapse(self, condition_split='block_pressure', 
                                  time_window=5, subject_level=True):
        """分析阈值塌陷（核心修复：解决reset_index列名冲突）"""
        def _sliding_wrapper(df):
            """包装滑动窗口函数，避免返回重复列"""
            res = self._sliding_window_analysis(df, window=time_window)
            # 移除分组列（避免与索引列冲突）
            if 'subj' in res.columns:
                res = res.drop(columns=['subj'])
            if 'condition' in res.columns:
                res = res.drop(columns=['condition'])
            return res
        
        # 分组计算滑动窗口（修复列名冲突）
        if subject_level:
            # 分组后reset_index(drop=True)，避免索引列与数据列重复
            windowed_data = self.data.groupby(['subj', 'condition']).apply(_sliding_wrapper)
            windowed_data = windowed_data.reset_index(drop=False)  # 保留分组索引为列
        else:
            windowed_data = self.data.groupby(['condition']).apply(_sliding_wrapper)
            windowed_data = windowed_data.reset_index(drop=False)
        
        # 拆分condition列为block_sign/block_pressure
        if 'condition' in windowed_data.columns:
            split_cols = windowed_data['condition'].str.split('_', expand=True).astype(str)
            if split_cols.shape[1] >= 2:
                windowed_data['block_sign'] = split_cols[0]
                def join_pressure(row):
                    parts = [p for p in row[1:] if p != 'nan']
                    return '_'.join(parts) if parts else 'unknown'
                windowed_data['block_pressure'] = split_cols.apply(join_pressure, axis=1)
        
        # 过滤NaN值后可视化
        windowed_data_valid = windowed_data.dropna(subset=['a'])
        if len(windowed_data_valid) == 0:
            print("警告：无有效数据用于阈值塌陷可视化")
            return windowed_data
        
        # 绘制阈值随试次变化的折线图
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=windowed_data_valid,
            x='window',
            y='a',
            hue=condition_split,
            style='block_sign',
            marker='o',
            ci=95
        )
        plt.title('决策阈值（边界分离）随试次的变化趋势', fontsize=14)
        plt.xlabel('试次窗口')
        plt.ylabel('决策阈值 (a)')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        plt.legend(title='实验条件')
        plt.tight_layout()
        plt.show()
        
        # 计算阈值塌陷程度（后期-前期）
        collapse_stats = self._calculate_collapse_magnitude(windowed_data_valid)
        print("\n阈值塌陷程度统计:")
        print(collapse_stats)
        
        return windowed_data
    
    def _sliding_window_analysis(self, df, window=5):
        """滑动窗口分析（单个被试+条件）"""
        results = []
        total_trials = len(df)
        df = df.sort_values('trial_num').copy()
        
        # 试次不足窗口大小则跳过
        if total_trials < window:
            print(f"警告：试次数({total_trials}) < 窗口大小({window})，跳过该组")
            return pd.DataFrame()
        
        # 滑动窗口计算EZ参数
        for i in range(0, total_trials - window + 1, window):
            window_df = df.iloc[i:i+window]
            window_num = i // window + 1
            
            # 计算窗口内的EZ统计量和参数
            stats = self._ez_stats(window_df)
            param_row = pd.Series({
                'window': window_num,
                'start_trial': i + 1,
                'end_trial': i + window,
                'n_trials': len(window_df)
            })
            param_row = pd.concat([param_row, stats])
            param_row = self._ez_calculate_params(param_row)
            
            results.append(param_row)
        
        return pd.DataFrame(results)
    
    def _calculate_collapse_magnitude(self, windowed_data):
        """量化阈值塌陷程度（后期阈值 - 前期阈值）"""
        first_window = windowed_data['window'].min()
        last_window = windowed_data['window'].max()
        
        # 提取前期/后期数据
        early = windowed_data[windowed_data['window'] == first_window][['subj', 'condition', 'a']].rename(columns={'a': 'a_early'})
        late = windowed_data[windowed_data['window'] == last_window][['subj', 'condition', 'a']].rename(columns={'a': 'a_late'})
        
        # 合并计算塌陷程度
        collapse = pd.merge(early, late, on=['subj', 'condition'], how='inner')
        collapse['collapse_magnitude'] = collapse['a_late'] - collapse['a_early']
        collapse['has_collapse'] = collapse['collapse_magnitude'] < 0  # 负值=阈值塌陷
        
        # 按条件汇总统计
        stats = collapse.groupby('condition').agg({
            'collapse_magnitude': ['mean', 'std', 'count'],
            'has_collapse': 'mean'  # 塌陷比例
        }).round(3)
        return stats
    
    def plot_param_comparison(self, param='a', split_by='block_pressure'):
        """可视化不同条件下的参数分布（箱线图）"""
        if self.params is None:
            raise ValueError("请先调用calculate_ez_parameters()计算参数")
        
        plot_data = self.params.dropna(subset=[param])
        if len(plot_data) == 0:
            print(f"警告：无有效数据用于{self._get_param_name(param)}可视化")
            return
        
        # 绘制箱线图
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=plot_data,
            x='block_sign',
            y=param,
            hue=split_by
        )
        plt.title(f'不同实验条件下的{self._get_param_name(param)}', fontsize=14)
        plt.xlabel('情境类型（gain/loss）')
        plt.ylabel(self._get_param_name(param))
        plt.legend(title='压力条件')
        plt.tight_layout()
        plt.show()
    
    def _get_param_name(self, param):
        """参数中文名称映射"""
        names = {'a': '决策阈值（边界分离）', 'v': '漂移率', 'ter': '非决策时间'}
        return names.get(param, param)

# 主程序运行
if __name__ == "__main__":
    # 替换为你的数据路径（单个CSV或目录）
    analyzer = EZDiffusionAnalyzer(data_path="./data")
    
    # 计算EZ参数
    params = analyzer.calculate_ez_parameters()
    print("\nEZ-Diffusion模型参数（过滤NaN后）:")
    print(params[['subj', 'condition', 'block_sign', 'block_pressure', 'a', 'v', 'ter']].round(3))
    
    # 可视化决策阈值的组间差异（箱线图）
    analyzer.plot_param_comparison(param='a')
    
    # 分析阈值塌陷（滑动窗口=5试次）
    collapse_data = analyzer.analyze_threshold_collapse(
        condition_split='block_pressure',
        time_window=5
    )