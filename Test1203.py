#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
项目名称：犹豫的轨迹 (Trajectory of Hesitation)
实验设计：2 (Sign: Gain/Loss) × 2 (Pressure: No/High)
核心技术：PEST 自适应 + 鼠标轨迹追踪 (Mouse Tracking) + DDM 数据流
适用场景：神经科学导论课程项目
"""

# --- 1. 导入依赖库 ---
from psychopy import gui, visual, core, data, event, logging, hardware
from psychopy.tools import environmenttools
import numpy as np
from numpy.random import random, shuffle, randint
import os
from datetime import datetime, timedelta

# --- 2. 基础设置 ---
# 确保路径正确
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# 实验信息弹窗
expName = 'Mouse_Track_Study'
expInfo = {
    'participant': f"{randint(0,999999):06.0f}", # 随机ID
    'age': '',
    'gender': ['M', 'F']
}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK: core.quit()

# 数据保存路径
dateStr = data.getDateStr()
filename = f"{_thisDir}/data/{expInfo['participant']}_{expName}_{dateStr}"

# 创建 ExperimentHandler (自动保存 .csv)
thisExp = data.ExperimentHandler(
    name=expName, version='2.0',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=_thisDir,
    savePickle=True, saveWideText=True,
    dataFileName=filename
)

# 创建窗口 (全屏, 黑色背景)
win = visual.Window(
    size=[1024, 768], fullscr=True, screen=0,
    winType='pyglet', color=[0,0,0], colorSpace='rgb',
    units='height' # 使用相对高度单位 (屏幕高=1.0)
)
win.mouseVisible = False # 初始隐藏鼠标

# 初始化鼠标对象
mouse = event.Mouse(win=win)

COLOR_PALETTE = {
    'deep_brown': '#92A501',      # 深蓝
    'light_blue': '#C5DFFA',      # 浅蓝
    'olive_green': '#AEB2D1',     #
    'orange': '#D989D4',          #紫色
    'light_gray': '#C9DCC4',      # 浅绿
    'gold': '#DAA87C',           # 褐色
    'cream': '#F4EEAC'           # 黄色
}

# --- 3. 视觉组件初始化 ---

# 3.1 布局组件 (三角形布局)
# 底部 Start 按钮区域
start_button = visual.Circle(win=win, radius=0.06, pos=(0, -0.4),
                             fillColor='grey', lineColor='white',lineWidth=4)
start_text = visual.TextStim(win=win, text="START", pos=(0, -0.4),
                             height=0.03, color='white', bold=True)

# 左上角选项区域 (Target A)
opt_left_box = visual.Rect(win=win, width=0.45, height=0.25, pos=(-0.4, 0.3),
                           fillColor=None, lineColor='white', lineWidth=6)
opt_left_txt = visual.TextStim(win=win, pos=(-0.4, 0.3), height=0.035, wrapWidth=0.4,bold=True)

# 右上角选项区域 (Target B)
opt_right_box = visual.Rect(win=win, width=0.45, height=0.25, pos=(0.4, 0.3),
                            fillColor=None, lineColor='white', lineWidth=6)
opt_right_txt = visual.TextStim(win=win, pos=(0.4, 0.3), height=0.035, wrapWidth=0.4,bold=True)

# 3.2 压力反馈组件
# 红色倒计时条 (放在屏幕中心或底部)
timer_bar = visual.Rect(win=win, width=1.2, height=0.02, pos=(0, 0),
                        fillColor='red', opacity=0) # 初始不可见

# 3.3 文本组件
intro_text = visual.TextStim(win=win, name='intro',
    text='【实验说明】\n\n我们将探究您在不同压力下的决策偏好。\n\n操作方式：\n1. 点击底部圆形的 START 按钮\n2. 迅速移动鼠标，点击左上或右上的选项\n\n包含【获得】与【损失】两种情境。\n遇到【红色倒计时】请务必加速！\n\n按空格键开始\n(按ESC键可随时退出实验)',
    font='SimHei', height=0.04, wrapWidth=1.4)

block_text = visual.TextStim(win=win, font='SimHei', height=0.05, wrapWidth=1.4)
end_text = visual.TextStim(win=win, text='实验结束\n数据已保存', font='SimHei', height=0.05)

# --- 新增：退出功能辅助函数 ---
def check_for_escape():
    """检查是否按下ESC键，如果按下则退出实验"""
    keys = event.getKeys(keyList=['escape'])
    if 'escape' in keys:
        # 显示退出提示
        quit_text = visual.TextStim(win=win, text='实验已中断\n按任意键退出', 
                                   font='SimHei', height=0.05, color='yellow')
        quit_text.draw()
        win.flip()
        core.wait(1)
        # 关闭窗口并退出
        win.close()
        core.quit()

# --- 4. 辅助函数：计算轨迹指标 ---
def calculate_metrics(traj_x, traj_y, start_pos, end_pos):
    """
    计算最大偏离度 (MD) - 衡量认知冲突的几何指标
    """
    if len(traj_x) < 2: return 0
    # 理想直线向量
    ideal_vec = np.array(end_pos) - np.array(start_pos)
    norm_ideal = np.linalg.norm(ideal_vec)
    if norm_ideal == 0: return 0

    max_dev = 0
    for x, y in zip(traj_x, traj_y):
        point_vec = np.array([x, y]) - np.array(start_pos)
        # 投影
        proj = np.dot(point_vec, ideal_vec) / norm_ideal
        # 垂直距离
        perp_dist = np.linalg.norm(point_vec - (proj * ideal_vec / norm_ideal))
        if perp_dist > max_dev:
            max_dev = perp_dist
    return max_dev

# --- 5. 实验主逻辑 ---

def run_experiment():
    # --- 5.1 实验设计矩阵 ---
    # 2 (Sign) x 2 (Pressure)
    blocks = []
    for s in ['gain', 'loss']:
        for p in ['no_pressure', 'high_pressure']:
            blocks.append({
                'sign': s,
                'pressure': p,
                'std_amount': 100,
                'std_days': 7
            })
    shuffle(blocks) # 随机顺序

    # 开始实验
    intro_text.draw()
    win.flip()
    
    # 修改等待按键的逻辑，添加ESC检查
    waiting_for_start = True
    while waiting_for_start:
        keys = event.getKeys(keyList=['space', 'escape'])
        if 'space' in keys:
            waiting_for_start = False
        elif 'escape' in keys:
            check_for_escape()
        # 轻微延迟以避免CPU过度使用
        core.wait(0.01)

    # --- 5.2 Block 循环 ---
    for block_idx, block in enumerate(blocks):
        curr_sign = block['sign']
        curr_pressure = block['pressure']

        # 设置视觉主题 (Theme)
        if curr_sign == 'gain':
            theme_color = '#FFD700' # 金色
            action = "获得"
            context_str = "【获得】情境"
        else:
            theme_color = '#CD5C5C' # 印度红
            action = "损失"
            context_str = "【损失】情境\n(假设拥有本金)"

        # 设置边框颜色
        opt_left_box.lineColor = theme_color
        opt_right_box.lineColor = theme_color
        start_button.lineColor = theme_color

        # 设置时间压力参数
        if curr_pressure == 'high_pressure':
            time_limit = 2.0 # 鼠标任务需要比按键稍长一点 (2.0s 是恐慌区)
            pressure_str = "【限时 2.0秒！】"
            bar_opacity = 1
        else:
            time_limit = 10.0
            pressure_str = "【不限时】"
            bar_opacity = 0

        # Block 提示
        block_text.text = f"第 {block_idx+1}/{len(blocks)} 组任务：\n\n{context_str}\n{pressure_str}\n\n按空格键开始\n(按ESC键可退出实验)"
        block_text.color = theme_color
        block_text.draw()
        win.flip()
        
        # 修改等待按键的逻辑，添加ESC检查
        waiting_for_block_start = True
        while waiting_for_block_start:
            keys = event.getKeys(keyList=['space', 'escape'])
            if 'space' in keys:
                waiting_for_block_start = False
            elif 'escape' in keys:
                check_for_escape()
            # 轻微延迟以避免CPU过度使用
            core.wait(0.01)

        # --- PEST 参数初始化 ---
        current_comp_m = 50 # 初始比较金额
        current_step = 25   # 初始步长
        min_step = 1.0
        pest_reversals = 0
        max_reversals = 6
        prev_choice_is_std = None

        # --- 5.3 Trial 循环 (PEST) ---
        trial_count = 0
        while pest_reversals < max_reversals:
            trial_count += 1
            
            # 检查ESC键
            check_for_escape()

            # --- A. 准备刺激文本 ---
            # 属性分离：时间在上，金额在下，中间空行 -> 验证属性潜伏期
            future_date = datetime.now() + timedelta(days=block['std_days'])
            date_str = f"{future_date.month}月{future_date.day}日"

            txt_std = f"{block['std_days']}天后 ({date_str})\n\n{action} {block['std_amount']}"
            txt_now = f"现在\n\n{action} {int(current_comp_m)}"

            # 随机位置 (左/右)
            if random() < 0.5:
                opt_left_txt.text = txt_std; opt_right_txt.text = txt_now
                standard_pos = 'left' # 标准选项在左
            else:
                opt_left_txt.text = txt_now; opt_right_txt.text = txt_std
                standard_pos = 'right' # 标准选项在右

            opt_left_txt.color = theme_color
            opt_right_txt.color = theme_color

            # --- B. 阶段1：归位 (Homing Phase) ---
            # 强制被试点击底部 Start 才能开始，确保每次轨迹起点一致
            mouse.setVisible(True)
            mouse.setPos((0, -0.4)) # 软复位
            clicked_start = False

            while not clicked_start:
                # 检查ESC键
                check_for_escape()
                
                # 绘制所有静态元素
                opt_left_box.draw(); opt_right_box.draw() # 空框
                start_button.draw(); start_text.draw()
                win.flip()

                # 检查是否点击 Start
                if mouse.getPressed()[0]:
                    if start_button.contains(mouse):
                        clicked_start = True

            # --- C. 阶段2：决策运动 (Motion Phase) ---
            trial_clock = core.Clock()
            traj_x = []
            traj_y = []

            has_responded = False
            choice_side = None # left or right
            rt = -1

            # 设置倒计时条
            timer_bar.opacity = bar_opacity

            while True:
                # 检查ESC键
                check_for_escape()
                
                t = trial_clock.getTime()

                # 1. 记录轨迹 (每帧)
                mx, my = mouse.getPos()
                traj_x.append(mx)
                traj_y.append(my)

                # 2. 超时检查
                if t > time_limit:
                    visual.TextStim(win=win, text="超时！太慢了！", color='red', height=0.08).draw()
                    win.flip()
                    core.wait(0.8)
                    break

                # 3. 绘制刺激
                opt_left_box.draw(); opt_left_txt.draw()
                opt_right_box.draw(); opt_right_txt.draw()

                # 绘制倒计时
                if timer_bar.opacity > 0:
                    remaining_ratio = max(0, (time_limit - t) / time_limit)
                    timer_bar.width = 1.2 * remaining_ratio
                    timer_bar.draw()

                win.flip()

                # 4. 检查点击
                if mouse.getPressed()[0]:
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
            if has_responded:
                # 判断选择
                chose_delayed = (choice_side == standard_pos)

                # 计算最大偏离度 (MD)
                start_p = (0, -0.4)
                # 终点坐标近似取框中心
                end_p = (-0.4, 0.3) if choice_side == 'left' else (0.4, 0.3)
                md = calculate_metrics(traj_x, traj_y, start_p, end_p)

                # 压缩轨迹数据为字符串 (x1,y1;x2,y2...) 以便存入CSV
                traj_str = ";".join([f"{x:.3f},{y:.3f}" for x,y in zip(traj_x, traj_y)])

                # 保存数据
                thisExp.addData('block_sign', curr_sign)
                thisExp.addData('block_pressure', curr_pressure)
                thisExp.addData('comp_m', current_comp_m)  # 难度指标
                thisExp.addData('rt', rt)
                thisExp.addData('chose_delayed', chose_delayed)
                thisExp.addData('max_deviation', md)       # 轨迹弯曲度
                thisExp.addData('raw_trajectory', traj_str)# 原始轨迹
                thisExp.addData('choice_side', choice_side)
                thisExp.nextEntry()

                # --- PEST 逻辑 (核心) ---
                # 1. 检查反转
                if prev_choice_is_std is not None:
                    if chose_delayed != prev_choice_is_std:
                        pest_reversals += 1
                        current_step = max(min_step, current_step / 2)

                # 2. 金额调整 (Gain与Loss相反)
                if curr_sign == 'gain':
                    # 获得: 选延迟 -> 增加即时金额(增加诱惑)
                    if chose_delayed: current_comp_m += current_step
                    else: current_comp_m -= current_step
                else:
                    # 损失: 选延迟损失 -> 减少即时损失(让即时没那么痛)
                    if chose_delayed: current_comp_m -= current_step
                    else: current_comp_m += current_step

                # 边界修正
                current_comp_m = max(1, min(99, current_comp_m))
                prev_choice_is_std = chose_delayed

            else:
                # 超时处理 (不更新PEST，记录空数据)
                thisExp.addData('block_sign', curr_sign)
                thisExp.addData('block_pressure', curr_pressure)
                thisExp.addData('rt', -1)
                thisExp.addData('chose_delayed', 'timeout')
                thisExp.nextEntry()

            # 试次间隔
            win.flip()
            core.wait(0.3)

    # --- 6. 结束 ---
    end_text.draw()
    win.flip()
    core.wait(2)
    print(f"\n数据已保存至: {filename}.csv\n")
    win.close()
    core.quit()

# 运行
if __name__ == '__main__':
    run_experiment()