import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from matplotlib.patches import Rectangle
from glob import glob

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class MouseTrajectoryAnalyzer:
    def __init__(self, data_path):
        """初始化分析器，加载数据"""
        self.data_path = data_path
        self.df = self.load_data(self.data_path)
        self.preprocess_data()

    def load_data(self, data_path):
        """加载实验数据"""
        if os.path.isdir(data_path):
            files = glob(os.path.join(data_path, "*.csv"))
            if not files:
                raise ValueError("指定目录中没有找到CSV文件")
            data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        else:
            data = pd.read_csv(data_path)
        
        print(f"成功加载数据，共{len(data)}条有效试次")
        return data
    
    def preprocess_data(self):
        """预处理数据，解析轨迹信息"""
        # 过滤超时试次
        self.df = self.df[self.df['chose_delayed'] != 'timeout'].copy()
        
        # 解析轨迹数据
        self.df['trajectory'] = self.df['raw_trajectory'].apply(
            lambda x: np.array([tuple(map(float, point.split(','))) 
                              for point in x.split(';')]) 
            if pd.notna(x) else None
        )
        
        # 计算额外的轨迹指标
        self.df['traj_length'] = self.df['trajectory'].apply(
            lambda traj: self.calculate_trajectory_length(traj) if traj is not None else None
        )
        
        self.df['curvature'] = self.df.apply(
            lambda row: self.calculate_curvature(row['trajectory'], row['choice_side']) 
            if row['trajectory'] is not None and pd.notna(row['choice_side']) else None, axis=1
        )
        
        self.df['initial_direction'] = self.df.apply(
            lambda row: self.calculate_initial_direction(row['trajectory'], row['choice_side']) 
            if row['trajectory'] is not None and pd.notna(row['choice_side']) else None, axis=1
        )
        
        self.df['direction_changes'] = self.df['trajectory'].apply(
            lambda traj: self.count_direction_changes(traj) if traj is not None else None
        )
        
        # 为双系统理论分析创建指标：冲突指数 = 最大偏离度 × 方向变化次数
        self.df['conflict_index'] = self.df['max_deviation'] * self.df['direction_changes']
        
        print("数据预处理完成")
    
    def calculate_trajectory_length(self, trajectory):
        """计算轨迹总长度"""
        if len(trajectory) < 2:
            return 0
        return np.sum(np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1)))
    
    def calculate_curvature(self, trajectory, choice_side):
        """计算轨迹曲率，衡量整体弯曲程度"""
        if len(trajectory) < 3:
            return 0
        
        # 起点和终点
        start_point = trajectory[0]
        end_point = trajectory[-1]
        
        # 计算理想直线距离
        straight_distance = np.linalg.norm(end_point - start_point)
        
        # 轨迹实际长度与直线距离的比值，作为曲率指标
        actual_length = self.calculate_trajectory_length(trajectory)
        return actual_length / straight_distance if straight_distance > 0 else 0
    
    def calculate_initial_direction(self, trajectory, choice_side):
        """计算初始方向与最终选择的一致性"""
        if len(trajectory) < 5:  # 需要足够的初始点
            return None
        
        # 初始方向向量（取前5个点）
        initial_vec = trajectory[4] - trajectory[0]
        
        # 目标方向
        target_x = -0.4 if choice_side == 'left' else 0.4
        target_vec = np.array([target_x, 0.3]) - trajectory[0]  # 终点坐标
        
        # 计算夹角余弦值，范围[-1,1]，值越大表示方向越一致
        dot_product = np.dot(initial_vec, target_vec)
        norm_initial = np.linalg.norm(initial_vec)
        norm_target = np.linalg.norm(target_vec)
        
        if norm_initial == 0 or norm_target == 0:
            return 0
            
        return dot_product / (norm_initial * norm_target)
    
    def count_direction_changes(self, trajectory):
        """计算水平方向变化次数，衡量犹豫程度"""
        if len(trajectory) < 3:
            return 0
        
        # 计算x方向的导数
        dx = np.diff(trajectory[:, 0])
        # 只关注显著的方向变化
        direction = np.sign(dx)
        direction[abs(dx) < 0.01] = 0
        
        # 计算方向变化次数
        changes = 0
        prev_dir = direction[0]
        for d in direction[1:]:
            if d != 0 and prev_dir != 0 and d != prev_dir:
                changes += 1
                prev_dir = d
            elif d != 0:
                prev_dir = d
                
        return changes
    
    def plot_average_trajectories(self, save_path=None):
        """按实验条件绘制平均轨迹"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        conditions = [
            ('gain', 'no_pressure'),
            ('gain', 'high_pressure'),
            ('loss', 'no_pressure'),
            ('loss', 'high_pressure')
        ]
        
        for i, (sign, pressure) in enumerate(conditions):
            ax = axes[i]
            condition_data = self.df[(self.df['block_sign'] == sign) & 
                                    (self.df['block_pressure'] == pressure)]
            
            # 分别绘制选择左侧和右侧的平均轨迹
            for choice_side in ['left', 'right']:
                choice_data = condition_data[condition_data['choice_side'] == choice_side]
                if len(choice_data) == 0:
                    continue
                
                # 对齐轨迹点: 插值到相同长度
                max_len = max(len(traj) for traj in choice_data['trajectory'])
                aligned_trajs = []
                
                for traj in choice_data['trajectory']:
                    x = np.linspace(0, 1, len(traj))
                    f_x = np.interp(np.linspace(0, 1, max_len), x, traj[:, 0])
                    f_y = np.interp(np.linspace(0, 1, max_len), x, traj[:, 1])
                    aligned_trajs.append(np.column_stack((f_x, f_y)))
                
                # 计算平均轨迹
                mean_traj = np.mean(aligned_trajs, axis=0)
                
                # 绘制
                ax.plot(mean_traj[:, 0], mean_traj[:, 1], 
                        label=f"选择{choice_side}", linewidth=2)
            
            # 绘制起点和目标区域
            ax.scatter(0, -0.4, color='black', s=100, label='起点')
            ax.add_patch(Rectangle((-0.625, 0.175), 0.45, 0.25, 
                                 fill=False, edgecolor='gray', linestyle='--'))
            ax.add_patch(Rectangle((0.175, 0.175), 0.45, 0.25, 
                                 fill=False, edgecolor='gray', linestyle='--'))
            
            ax.set_title(f"{sign}情境 - {pressure}")
            ax.set_xlim(-0.8, 0.8)
            ax.set_ylim(-0.5, 0.6)
            ax.set_aspect('equal')
            ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"平均轨迹图已保存至 {save_path}")
        plt.show()
    
    def analyze_by_condition(self):
        """按实验条件分析轨迹指标"""
        # 定义要分析的指标
        metrics = ['rt', 'max_deviation', 'curvature', 
                  'direction_changes', 'conflict_index', 'initial_direction']
        
        # 按条件分组计算均值和标准差
        results = self.df.groupby(['block_sign', 'block_pressure'])[metrics].agg(['mean', 'std'])
        print("\n各条件下的轨迹指标均值和标准差：")
        print(results.round(4))
        
        # 双因素方差分析：检验sign和pressure的主效应及交互作用
        print("\n双因素方差分析结果：")
        for metric in metrics:
            print(f"\n指标: {metric}")
            sign_effect = stats.f_oneway(
                self.df[self.df['block_sign'] == 'gain'][metric].dropna(),
                self.df[self.df['block_sign'] == 'loss'][metric].dropna()
            )
            pressure_effect = stats.f_oneway(
                self.df[self.df['block_pressure'] == 'no_pressure'][metric].dropna(),
                self.df[self.df['block_pressure'] == 'high_pressure'][metric].dropna()
            )
            
            # 简单效应分析
            gain_no = self.df[(self.df['block_sign'] == 'gain') & 
                             (self.df['block_pressure'] == 'no_pressure')][metric].dropna()
            gain_high = self.df[(self.df['block_sign'] == 'gain') & 
                              (self.df['block_pressure'] == 'high_pressure')][metric].dropna()
            loss_no = self.df[(self.df['block_sign'] == 'loss') & 
                             (self.df['block_pressure'] == 'no_pressure')][metric].dropna()
            loss_high = self.df[(self.df['block_sign'] == 'loss') & 
                              (self.df['block_pressure'] == 'high_pressure')][metric].dropna()
            
            interaction = stats.f_oneway(gain_no, gain_high, loss_no, loss_high)
            
            print(f"  Sign主效应: F={sign_effect.statistic:.3f}, p={sign_effect.pvalue:.4f}")
            print(f"  Pressure主效应: F={pressure_effect.statistic:.3f}, p={pressure_effect.pvalue:.12f}")
            print(f"  交互效应: F={interaction.statistic:.3f}, p={interaction.pvalue:.8f}")
    
    def plot_metric_comparison(self, metric, save_path=None):
        """绘制不同条件下特定指标的比较图"""
        plt.figure(figsize=(10, 6))
        sns.barplot(x='block_sign', y=metric, hue='block_pressure', data=self.df)
        plt.title(f'{metric}在不同实验条件下的比较')
        plt.xlabel('情境类型')
        plt.ylabel(metric)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"{metric}比较图已保存至 {save_path}")
        plt.show()
    
    def plot_conflict_vs_rt(self, save_path=None):
        """绘制冲突指数与反应时的关系，检验双系统理论预测"""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='rt', y='conflict_index', 
                       hue='block_sign', style='block_pressure',
                       data=self.df)
        plt.title('反应时与冲突指数的关系')
        plt.xlabel('反应时 (秒)')
        plt.ylabel('冲突指数 (最大偏离度 × 方向变化次数)')
        
        # 计算并绘制回归线
        for sign in ['gain', 'loss']:
            for pressure in ['no_pressure', 'high_pressure']:
                subset = self.df[(self.df['block_sign'] == sign) & 
                                (self.df['block_pressure'] == pressure)]
                if len(subset) > 10:  # 需要足够数据点
                    z = np.polyfit(subset['rt'], subset['conflict_index'], 1)
                    p = np.poly1d(z)
                    plt.plot(subset['rt'], p(subset['rt']), 
                             label=f"{sign} {pressure}")
        
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"冲突指数与反应时关系图已保存至 {save_path}")
        plt.show()


if __name__ == "__main__":
    analyzer = MouseTrajectoryAnalyzer(data_path="./data")
    
    analyzer.plot_average_trajectories("average_trajectories.png")
    
    analyzer.analyze_by_condition()
    
    analyzer.plot_metric_comparison("conflict_index", "conflict_index_comparison.png")
    analyzer.plot_metric_comparison("max_deviation", "max_deviation_comparison.png")
    
    analyzer.plot_conflict_vs_rt("conflict_vs_rt.png")
    
