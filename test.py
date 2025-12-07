#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding
from datetime import datetime, timedelta
from psychopy.hardware import keyboard

# --- Setup global variables ---
deviceManager = hardware.DeviceManager()
_thisDir = os.path.dirname(os.path.abspath(__file__))
psychopyVersion = '2025.1.1'
expName = 'temporal_discounting_load'
expVersion = ''
runAtExit = []
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Pilot mode setup ---
PILOTING = core.setPilotModeFromArgs()
_fullScr = True
_winSize = (1024, 768)
if PILOTING:
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        _winSize = prefs.piloting['forcedWindowSize']
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True)
    if dlg.OK == False:
        core.quit()
    return expInfo

def setupData(expInfo, dataDir=None):
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath=_thisDir,
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    return thisExp

def setupLogging(filename):
    if PILOTING:
        logging.console.setLevel(prefs.piloting['pilotConsoleLoggingLevel'])
    else:
        logging.console.setLevel('warning')
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(prefs.piloting['pilotLoggingLevel'])
    else:
        logFile.setLevel(logging.getLevel('info'))
    return logFile

def setupWindow(expInfo=None, win=None):
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    if win is None:
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False
        )
    else:
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    return win

def setupDevices(expInfo, thisExp, win):
    deviceManager.ioServer = None
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb')
    if deviceManager.getDevice('intro_key_resp') is None:
        deviceManager.addDevice(deviceClass='keyboard', deviceName='intro_key_resp')
    if deviceManager.getDevice('trial_key_resp') is None:
        deviceManager.addDevice(deviceClass='keyboard', deviceName='trial_key_resp')
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    if thisExp.status != PAUSED:
        return
    pauseTimer = core.Clock()
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    while thisExp.status == PAUSED:
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        clock.time.sleep(0.001)
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())

def endExperiment(thisExp, win=None):
    if win is not None:
        win.clearAutoDraw()
        win.flip()
    logging.console.setLevel(logging.WARNING)
    thisExp.status = FINISHED
    for fcn in runAtExit:
        fcn()
    logging.flush()

def quit(thisExp, win=None, thisSession=None):
    thisExp.abort()
    if win is not None:
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    core.quit()

# --- Main Run Function ---
def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    thisExp.status = STARTED
    win.winHandle.activate()
    exec = environmenttools.setExecEnvironment(globals())
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    
    # --- Experiment Parameters Setup (创新点核心配置) ---
    experiment_start_date = datetime.now()

    # 1. 定义实验条件 (Block Design)
    # 为了防止时间过长，我们只选 3 个关键时间点，但增加金额和负载维度
    # 维度：2(金额) x 2(负载) x 3(延迟) = 12个Block
    
    amounts = [100, 10000]       # 创新点1：小金额 vs 大金额
    loads = ['low', 'high']      # 创新点2：认知负载 (低:1位数字, 高:7位数字)
    delays = [7, 30, 365]        # 延迟天数 (1周, 1月, 1年)

    # 生成所有Block的组合
    blocks = []
    for amt in amounts:
        for ld in loads:
            for d in delays:
                blocks.append({
                    'amount': amt,
                    'load': ld,
                    'delay': d
                })
    # 打乱Block顺序 (Block Randomization)
    shuffle(blocks)

    # --- Initialize Components ---
    
    # Intro Routine
    intro_text = visual.TextStim(win=win, name='intro_text',
        text='欢迎参与实验！\n\n本实验包含多个部分，涉及决策与记忆任务。\n\n每个试次开始前，屏幕会出现一组数字，请务必【记住它】。\n之后会出现两个选项，请按左右键选择偏好：\n左侧："立即获得..."   右侧："X天后获得..."\n\n选择完成后，请回忆并核对刚才的数字。\n\n按"空格"键开始实验',
        font='SimHei', pos=(0, 0), height=0.04, wrapWidth=1.5, color='white')
    intro_key_resp = keyboard.Keyboard(deviceName='intro_key_resp')
    
    # Memory Routine Components (认知负载)
    mem_text = visual.TextStim(win=win, name='mem_text',
        text='', font='Arial', pos=(0, 0), height=0.08, color='yellow')
    
    # Trial Routine Components
    trial_fixation = visual.TextStim(win=win, text='+', font='Arial', height=0.05, pos=(0,0))
    # 调整了 wrapWidth 和 height 防止文字被挡
    trial_opt1 = visual.TextStim(win=win, text='', font='SimHei', pos=(-0.4, 0), height=0.035, wrapWidth=0.7)
    trial_opt2 = visual.TextStim(win=win, text='', font='SimHei', pos=(0.4, 0), height=0.035, wrapWidth=0.7)
    trial_key_resp = keyboard.Keyboard(deviceName='trial_key_resp')

    # Recall/Check Routine Components
    recall_text = visual.TextStim(win=win, text='', font='SimHei', pos=(0, 0.1), height=0.04, wrapWidth=1.5)
    reveal_text = visual.TextStim(win=win, text='', font='Arial', pos=(0, -0.1), height=0.06, color='yellow')
    recall_key_resp = keyboard.Keyboard()

    # End Routine
    end_text = visual.TextStim(win=win, text='实验结束！感谢参与！', font='SimHei', height=0.05)

    # --- Timers ---
    if globalClock is None: globalClock = core.Clock()
    routineTimer = core.Clock()
    
    # --- Run Intro ---
    # (Intro code omitted for brevity, logic same as before but using updated intro_text)
    # 简写 Intro 运行逻辑
    intro_text.draw()
    win.flip()
    event.waitKeys(keyList=['space'])

    # --- Loop through Blocks ---
    for block in blocks:
        # 提取当前Block的条件
        curr_amount = block['amount']
        curr_load = block['load']
        curr_delay = block['delay']

        # Block间休息与提示
        block_info = f"下一组任务：\n记忆难度：{'【简单 (1位)】' if curr_load == 'low' else '【困难 (7位)】'}\n金额范围：{curr_amount} 元\n\n按空格键开始"
        rest_text = visual.TextStim(win=win, text=block_info, font='SimHei', height=0.04, wrapWidth=1.5)
        rest_text.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
        
        # 休息缓冲
        win.clearBuffer()
        win.flip()
        core.wait(1.0)

        # --- PEST 参数动态设置 (针对大金额的调整) ---
        # 如果是10000元，初始比较金额设为5000，步长2500
        # 如果是100元，初始比较金额50，步长25
        if curr_amount == 10000:
            current_comp_m = 5000
            current_step = 2500
            min_step_limit = 50 # 精度控制
        else:
            current_comp_m = 50
            current_step = 25
            min_step_limit = 1

        pest_reversals = 0
        max_reversals = 6
        prev_choice_is_standard = None
        
        # --- Trial Loop (PEST) ---
        while pest_reversals < max_reversals:
            
            # --- 1. 生成记忆材料 (Cognitive Load) ---
            if curr_load == 'high':
                # 高负载：7位随机数字
                mem_digits = "".join([str(randint(0, 9)) for _ in range(7)])
                show_time = 4.0 # 给多一点时间记忆
            else:
                # 低负载：1位随机数字
                mem_digits = str(randint(0, 9))
                show_time = 1.5
            
            # 呈现记忆数字
            mem_text.text = mem_digits
            mem_text.draw()
            win.flip()
            core.wait(show_time)
            
            # 掩蔽/保持阶段 (Blank screen)
            win.clearBuffer()
            win.flip()
            core.wait(0.5) # 0.5秒保持期

            # --- 2. 准备跨期选择材料 ---
            # 计算日期
            future_date = experiment_start_date + timedelta(days=curr_delay)
            try:
                date_str = future_date.strftime("%m月%d日")
                year_str = future_date.strftime("%Y年")
            except:
                date_str = f"{future_date.month}月{future_date.day}日"
                year_str = f"{future_date.year}年"

            # 生成选项文本 (Date Framing)
            if random() < 0.5:
                # 左：延迟(标准)  右：即时(比较)
                opt1_text = f"{year_str}{date_str}\n获得 {curr_amount} 元\n(现在得0)"
                opt2_text = f"现在\n获得 {current_comp_m} 元\n({year_str}{date_str}得0)"
                standard_side = 'left'
            else:
                opt1_text = f"现在\n获得 {current_comp_m} 元\n({year_str}{date_str}得0)"
                opt2_text = f"{year_str}{date_str}\n获得 {curr_amount} 元\n(现在得0)"
                standard_side = 'right'

            # --- 3. 运行选择 (Choice Phase) ---
            trial_opt1.text = opt1_text
            trial_opt2.text = opt2_text
            
            # 绘制注视点
            trial_fixation.draw()
            win.flip()
            core.wait(0.5)
            
            # 绘制选项并监听按键
            trial_opt1.draw()
            trial_opt2.draw()
            win.flip()
            
            # 记录反应时 RT
            kb_clock = core.Clock()
            keys = trial_key_resp.waitKeys(keyList=['left', 'right', 'escape'])
            rt = kb_clock.getTime()
            
            if 'escape' in keys:
                endExperiment(thisExp, win=win)
                return

            resp_key = keys[0]
            
            # --- 4. 回忆/核对阶段 (Recall Phase) ---
            # 简单起见，让被试按空格查看答案并自我核对
            # 这保持了认知负载，同时避免了复杂的输入编程
            recall_text.text = "刚才的数字是？\n（请在脑中回忆，然后按空格键查看答案）"
            recall_text.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
            
            reveal_text.text = mem_digits
            recall_text.text = "正确答案是："
            recall_text.draw()
            reveal_text.draw()
            # 下方加个继续提示
            visual.TextStim(win=win, text='按空格键继续', pos=(0,-0.3), height=0.03).draw()
            win.flip()
            event.waitKeys(keyList=['space'])

            # --- 5. 数据记录与PEST更新 ---
            
            # 记录关键变量
            thisExp.addData('block_amount', curr_amount)      # 创新点1：金额条件
            thisExp.addData('block_load', curr_load)          # 创新点2：负载条件
            thisExp.addData('block_delay', curr_delay)
            thisExp.addData('memory_digits', mem_digits)
            thisExp.addData('comp_m', current_comp_m)
            thisExp.addData('choice', resp_key)
            thisExp.addData('rt', rt)                         # 创新点3：反应时
            
            # 判断选择
            is_standard = (resp_key == standard_side)
            thisExp.addData('chose_delayed', is_standard)
            thisExp.nextEntry()

            # PEST 逻辑更新
            if prev_choice_is_standard is not None:
                if is_standard != prev_choice_is_standard:
                    pest_reversals += 1
                    current_step = max(current_step / 2, min_step_limit)
            
            if is_standard:
                # 选了延迟(标准)，说明即时金额太低 -> 加钱
                current_comp_m += current_step
            else:
                # 选了即时，说明即时金额太高 -> 减钱
                current_comp_m -= current_step
            
            current_comp_m = int(max(1, current_comp_m)) # 保持整数
            prev_choice_is_standard = is_standard

    # --- End Experiment ---
    end_text.draw()
    win.flip()
    core.wait(3)
    endExperiment(thisExp, win=win)

if __name__ == '__main__':
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(expInfo=expInfo, thisExp=thisExp, win=win, globalClock='float')
    quit(thisExp=thisExp, win=win)