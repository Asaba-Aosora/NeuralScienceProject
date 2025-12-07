#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
项目名称：犹豫的轨迹 (Trajectory of Hesitation)
文件：experiment_main.py
功能：实验执行、刺激呈现、PEST自适应控制、鼠标轨迹数据采集与保存。
"""

# --- 1. 导入依赖库 ---
from psychopy import gui, visual, core, data, event
import numpy as np
from numpy.random import random, shuffle, randint
import os
from datetime import datetime, timedelta

# --- 2. 基础设置 ---
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# 实验信息弹窗
expName = 'Mouse_Track_Study'
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",  # 随机ID
    'age': '',
    'gender': ['M', 'F']
}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK: core.quit()

# 数据保存路径：确保路径是 /data/
dateStr = data.getDateStr()
filename = f"{_thisDir}{os.sep}data{os.sep}{expInfo['participant']}_{expName}_{dateStr}"

# 确保数据文件夹存在
if not os.path.exists(_thisDir + os.sep + 'data'):
    os.makedirs(_thisDir + os.sep + 'data')

# 创建 ExperimentHandler (自动保存.csv 和.psydat)
thisExp = data.ExperimentHandler(
    name=expName, version='2.1',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=_thisDir,
    savePickle=True, saveWideText=True,
    dataFileName=filename
)

# 创建窗口
win = visual.Window(
    size=[1024, 768], fullscr=True, screen=0,
    winType='pyglet', color=[0,0,0], colorSpace='rgb',
    units='height'
)
win.mouseVisible = False

# 初始化鼠标对象
mouse = event.Mouse(win=win)

# --- 3. 视觉组件初始化 ---

# 3.1 布局组件
start_button = visual.Circle(win=win, radius=0.06, pos=(0, -0.4),
                             fillColor='grey', lineColor='white')
start_text = visual.TextStim(win=win, text="START", pos=(0, -0.4),
                             height=0.03, color='white', bold=True)

opt_left_box = visual.Rect(win=win, width=0.45, height=0.25, pos=(-0.4, 0.3),
                           fillColor=None, lineColor='white', lineWidth=3)
opt_left_txt = visual.TextStim(win=win, pos=(-0.4, 0.3), height=0.035, wrapWidth=0.4)

opt_right_box = visual.Rect(win=win, width=0.45, height=0.25, pos=(0.4, 0.3),
                            fillColor=None, lineColor='white', lineWidth=3)
opt_right_txt = visual.TextStim(win=win, pos=(0.4, 0.3), height=0.035, wrapWidth=0.4)

# 3.2 压力反馈组件
timer_bar = visual.Rect(win=win, width=1.2, height=0.02, pos=(0, 0),
                        fillColor='red', opacity=0, autoLog=False)

# 3.3 文本组件
intro_text = visual.TextStim(win=win, name='intro',
                             text='【实验说明】\n\n我们将探究您在不同压力下的决策偏好。\n\n操作方式：\n1. 点击底部圆形的 START 按钮\n2. 迅速移动鼠标，点击左上或右上的选项\n\n包含【获得】与【损失】两种情境。\n遇到【红色倒计时】请务必加速！\n\n按空格键开始',
                             font='SimHei', height=0.04, wrapWidth=1.4)

block_text = visual.TextStim(win=win, font='SimHei', height=0.05, wrapWidth=1.4)
feedback_text = visual.TextStim(win=win, text="", color='red', height=0.05)
end_text = visual.TextStim(win=win, text='实验结束\n数据已保存', font='SimHei', height=0.05)


# --- 4. 辅助函数：计算轨迹指标 ---
def calculate_metrics(traj_x, traj_y, start_pos, end_pos):
    """
    计算最大偏离度 (MD) - 衡量认知冲突的几何指标
    """
    if len(traj_x) < 2: return 0
    # 转换为 NumPy 数组
    traj_points = np.array(list(zip(traj_x, traj_y)))
    start_p = np.array(start_pos)
    end_p = np.array(end_pos)

    # 理想直线向量
    ideal_vec = end_p - start_p
    norm_ideal = np.linalg.norm(ideal_vec)

    if norm_ideal == 0: return 0

    max_dev = 0

    # 遍历轨迹点，计算到直线的垂直距离
    for point_vec in traj_points - start_p:
        # 向量投影：投影长度 / 理想向量长度
        proj_len = np.dot(point_vec, ideal_vec) / norm_ideal
        # 投影点：起点 + 投影长度 * 单位理想向量
        projection_point = start_p + proj_len * (ideal_vec / norm_ideal)

        # 垂直距离
        perp_dist = np.linalg.norm(point_vec - (projection_point - start_p))
        if perp_dist > max_dev:
            max_dev = perp_dist

    return max_dev


# --- 5. 实验主逻辑 ---

def run_experiment():
    # --- 5.1 实验设计矩阵 ---
    blocks = []
    # 2 (Sign) x 2 (Pressure) = 4 conditions
    for s in ['gain', 'loss']:
        for p in ['no_pressure', 'high_pressure']:
            blocks.append({
                'sign': s,
                'pressure': p,
                'std_amount': 100,  # A = 100
                'std_days': 7  # D = 7
            })
    shuffle(blocks)

    # 引导页
    intro_text.draw()
    win.flip()
    event.waitKeys(keyList=['space'])

    # --- 5.2 Block 循环 ---
    for block in blocks:
        curr_sign = block['sign']
        curr_pressure = block['pressure']

        # 设置主题
        theme_color = '#FFD700' if curr_sign == 'gain' else '#CD5C5C'
        action = "获得" if curr_sign == 'gain' else "损失"
        context_str = f"【{action}】情境" if curr_sign == 'gain' else f"【{action}】情境\n(假设拥有本金)"

        opt_left_box.lineColor = theme_color
        opt_right_box.lineColor = theme_color
        start_button.lineColor = theme_color

        # 设置时间压力参数
        if curr_pressure == 'high_pressure':
            time_limit = 2.0
            pressure_str = "【限时 2.0秒！】"
            bar_opacity = 1
        else:
            time_limit = 10.0  # 实际上是自由时，给一个宽松上限
            pressure_str = "【不限时】"
            bar_opacity = 0

        # Block 提示
        block_text.text = f"下一组任务：\n\n{context_str}\n{pressure_str}\n\n按空格键开始"
        block_text.color = theme_color
        block_text.draw()
        win.flip()
        event.waitKeys(keyList=['space'])

        # --- PEST 参数初始化 ---
        current_comp_m = 50.0  # 初始比较金额
        current_step = 25.0  # 初始步长
        min_step = 1.0
        pest_reversals = 0
        max_reversals = 3  # 稍微增加到3次反转，提高收敛速度和效率
        prev_choice_is_std = None  # 记录上一次的选择是否是延迟/标准选项

        # --- 5.3 Trial 循环 (PEST) ---
        trial_index = 0
        while pest_reversals < max_reversals:
            trial_index += 1

            # --- A. 准备刺激文本 ---
            future_date = datetime.now() + timedelta(days=block['std_days'])
            date_str = f"{future_date.month}月{future_date.day}日"

            # 时间在上，金额在下（用于属性潜伏期分析）
            txt_std = f"{block['std_days']}天后 ({date_str})\n\n{action} {block['std_amount']:.0f}"
            txt_now = f"现在\n\n{action} {current_comp_m:.2f}"  # 保留两位小数显示

            # 随机位置
            if random() < 0.5:
                opt_left_txt.text = txt_std;
                opt_right_txt.text = txt_now
                standard_pos = 'left'
            else:
                opt_left_txt.text = txt_now;
                opt_right_txt.text = txt_std
                standard_pos = 'right'

            opt_left_txt.color = theme_color
            opt_right_txt.color = theme_color

            # --- B. 阶段1：归位 (Homing Phase) ---
            mouse.setVisible(True)
            # 强制被试将鼠标移回起点 (0, -0.4) 附近
            mouse.setPos((0, -0.4))
            clicked_start = False

            # 绘制阶段：只画起点和空框，不显示选项内容
            while not clicked_start:
                opt_left_box.draw();
                opt_right_box.draw()  # 空框
                start_button.draw();
                start_text.draw()
                win.flip()

                if mouse.getPressed():
                    if start_button.contains(mouse):
                        clicked_start = True

            # --- C. 阶段2：决策运动 (Motion Phase) ---
            trial_clock = core.Clock()
            traj_x = []
            traj_y = []

            has_responded = False
            choice_side = None
            rt = -1

            timer_bar.opacity = bar_opacity
            mouse.clickReset()  # 清除开始按钮的点击状态

            while True:
                t = trial_clock.getTime()

                # 1. 记录轨迹 (每帧)
                mx, my = mouse.getPos()
                traj_x.append(mx)
                traj_y.append(my)

                # 2. 超时检查
                if t > time_limit:
                    feedback_text.text = "超时！太慢了！"
                    feedback_text.draw()
                    win.flip()
                    has_responded = False  # 标记为未响应
                    core.wait(0.8)
                    break

                # 3. 绘制刺激
                opt_left_box.draw();
                opt_left_txt.draw()
                opt_right_box.draw();
                opt_right_txt.draw()

                # 绘制倒计时
                if timer_bar.opacity > 0:
                    remaining_ratio = max(0, (time_limit - t) / time_limit)
                    timer_bar.width = 1.2 * remaining_ratio
                    timer_bar.draw()

                win.flip()

                # 4. 检查点击
                if mouse.getPressed():
                    if opt_left_box.contains(mouse):
                        choice_side = 'left'
                        rt = t
                        has_responded = True
                        break
                    elif opt_right_box.contains(mouse):
                        choice_side = 'right'
                        rt = t
                        has_responded = True
                        break

            # --- D. 数据记录与算法更新 ---

            # 轨迹数据压缩
            traj_str = ";".join([f"{x:.3f},{y:.3f}" for x, y in zip(traj_x, traj_y)])

            if has_responded:
                # 判断选择
                chose_delayed = (choice_side == standard_pos)

                # 计算最大偏离度 (MD)
                start_p = (0.0, -0.4)
                # 终点坐标近似取框中心
                end_p = (-0.4, 0.3) if choice_side == 'left' else (0.4, 0.3)
                md = calculate_metrics(traj_x, traj_y, start_p, end_p)

                # 保存试次数据
                thisExp.addData('trial_number', trial_index)
                thisExp.addData('block_sign', curr_sign)
                thisExp.addData('block_pressure', curr_pressure)
                thisExp.addData('comp_m', current_comp_m)  # 即时金额 (M)
                thisExp.addData('std_amount', block['std_amount'])
                thisExp.addData('std_days', block['std_days'])
                thisExp.addData('standard_pos', standard_pos)
                thisExp.addData('choice_side', choice_side)
                thisExp.addData('rt', rt)
                thisExp.addData('chose_delayed', chose_delayed)  # True/False
                thisExp.addData('max_deviation', md)
                thisExp.addData('raw_trajectory', traj_str)
                thisExp.addData('time_limit', time_limit)

                # PEST 逻辑 (核心)
                if prev_choice_is_std is not None:
                    # 检查反转
                    if chose_delayed != prev_choice_is_std:
                        pest_reversals += 1
                        current_step = max(min_step, current_step / 2.0)  # 步长减半

                # 金额调整 (Gain与Loss相反)
                if curr_sign == 'gain':
                    # 获得: 选延迟 -> 增加即时金额(增加诱惑)
                    if chose_delayed:
                        current_comp_m += current_step
                    else:
                        current_comp_m -= current_step
                else:
                    # 损失: 选延迟损失 -> 减少即时损失(让即时没那么痛)
                    if chose_delayed:
                        current_comp_m -= current_step
                    else:
                        current_comp_m += current_step

                # 边界修正
                current_comp_m = max(1.0, min(block['std_amount'] * 2, current_comp_m))  # 金额不能低于1
                prev_choice_is_std = chose_delayed

                # 记录 PEST 状态
                thisExp.addData('pest_reversals_at_end', pest_reversals)
                thisExp.addData('current_step_used', current_step)
                thisExp.nextEntry()

            else:
                # 超时处理 (不更新PEST，记录空数据)
                thisExp.addData('trial_number', trial_index)
                thisExp.addData('block_sign', curr_sign)
                thisExp.addData('block_pressure', curr_pressure)
                thisExp.addData('comp_m', current_comp_m)
                thisExp.addData('rt', time_limit)  # 超时RT设为上限
                thisExp.addData('chose_delayed', 'timeout')
                thisExp.addData('max_deviation', np.nan)
                thisExp.addData('raw_trajectory', 'timeout')
                thisExp.addData('time_limit', time_limit)
                thisExp.addData('pest_reversals_at_end', pest_reversals)
                thisExp.addData('current_step_used', current_step)
                thisExp.nextEntry()

            # 试次间隔
            win.flip()
            core.wait(0.3)

    # --- 6. 结束 ---
    thisExp.saveAsWideText(filename + '.csv', appendDate=False)
    thisExp.close()
    end_text.draw()
    win.flip()
    core.wait(2)
    win.close()
    core.quit()


# 运行
if __name__ == '__main__':
    run_experiment()