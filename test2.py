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

import numpy as np
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os
import sys
from datetime import datetime, timedelta
from psychopy.hardware import keyboard

# --- Setup global variables ---
deviceManager = hardware.DeviceManager()
_thisDir = os.path.dirname(os.path.abspath(__file__))
psychopyVersion = '2025.1.1'
expName = 'fast_test_demo'  # 修改实验名，方便区分
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

    # 关键：这里确保保存 .csv (saveWideText=True)
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
    logFile = logging.LogFile(filename + '.log')
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
            monitor='testMonitor', color=[0, 0, 0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False
        )
    else:
        win.color = [0, 0, 0]
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

    # --- Experiment Parameters Setup (3分钟快速测试版) ---
    experiment_start_date = datetime.now()

    # 【修改1】极简参数设置：总共 2 个Block，测试 1 种金额，2 种负载，1 个延迟
    amounts = [100]  # 仅测试 100元
    loads = ['low', 'high']  # 测试 2 种负载 (确保创新点逻辑跑通)
    delays = [7]  # 仅测试 7天 (减少循环)

    # 生成 Block 列表 (共 1 * 2 * 1 = 2 个Block)
    blocks = []
    for amt in amounts:
        for ld in loads:
            for d in delays:
                blocks.append({
                    'amount': amt,
                    'load': ld,
                    'delay': d
                })
    shuffle(blocks)  # 打乱顺序

    # --- Initialize Components ---
    intro_text = visual.TextStim(win=win, name='intro_text',
                                 text='【快速测试模式】\n\n我们将进行2组简短测试。\n请记住屏幕上的数字，并完成选择。\n\n按"空格"键开始',
                                 font='SimHei', pos=(0, 0), height=0.04, wrapWidth=1.5, color='white')
    intro_key_resp = keyboard.Keyboard(deviceName='intro_key_resp')

    mem_text = visual.TextStim(win=win, name='mem_text',
                               text='', font='Arial', pos=(0, 0), height=0.08, color='yellow')

    trial_fixation = visual.TextStim(win=win, text='+', font='Arial', height=0.05, pos=(0, 0))
    trial_opt1 = visual.TextStim(win=win, text='', font='SimHei', pos=(-0.4, 0), height=0.035, wrapWidth=0.7)
    trial_opt2 = visual.TextStim(win=win, text='', font='SimHei', pos=(0.4, 0), height=0.035, wrapWidth=0.7)
    trial_key_resp = keyboard.Keyboard(deviceName='trial_key_resp')

    recall_text = visual.TextStim(win=win, text='', font='SimHei', pos=(0, 0.1), height=0.04, wrapWidth=1.5)
    reveal_text = visual.TextStim(win=win, text='', font='Arial', pos=(0, -0.1), height=0.06, color='yellow')

    end_text = visual.TextStim(win=win, text='测试结束！\n数据已保存', font='SimHei', height=0.05)

    if globalClock is None: globalClock = core.Clock()
    routineTimer = core.Clock()

    # --- Run Intro ---
    intro_text.draw()
    win.flip()
    event.waitKeys(keyList=['space'])

    # --- Loop through Blocks ---
    for block in blocks:
        curr_amount = block['amount']
        curr_load = block['load']
        curr_delay = block['delay']

        # Block间简短提示
        block_info = f"任务：{'简单(1位)' if curr_load == 'low' else '困难(7位)'}\n金额：{curr_amount}元\n按空格开始"
        rest_text = visual.TextStim(win=win, text=block_info, font='SimHei', height=0.04, wrapWidth=1.5)
        rest_text.draw()
        win.flip()
        event.waitKeys(keyList=['space'])

        win.clearBuffer()
        win.flip()
        core.wait(0.5)  # 缩短休息时间

        # --- PEST 参数 ---
        current_comp_m = 50
        current_step = 25
        min_step_limit = 1

        pest_reversals = 0
        # 【修改2】最大反转次数改为2，极大缩短每个Block的长度
        max_reversals = 2
        prev_choice_is_standard = None

        # --- Trial Loop ---
        while pest_reversals < max_reversals:

            # 1. 记忆材料
            if curr_load == 'high':
                mem_digits = "".join([str(randint(0, 9)) for _ in range(7)])
                show_time = 2.0  # 【修改3】缩短记忆呈现时间
            else:
                mem_digits = str(randint(0, 9))
                show_time = 1.0

            mem_text.text = mem_digits
            mem_text.draw()
            win.flip()
            core.wait(show_time)

            win.clearBuffer()
            win.flip()
            core.wait(0.2)  # 缩短保持期

            # 2. 跨期选择材料
            future_date = experiment_start_date + timedelta(days=curr_delay)
            try:
                date_str = future_date.strftime("%m月%d日")
                year_str = future_date.strftime("%Y年")
            except:
                date_str = f"{future_date.month}月{future_date.day}日"
                year_str = f"{future_date.year}年"

            if random() < 0.5:
                opt1_text = f"{year_str}{date_str}\n获得 {curr_amount} 元\n(现在得0)"
                opt2_text = f"现在\n获得 {current_comp_m} 元\n({year_str}{date_str}得0)"
                standard_side = 'left'
            else:
                opt1_text = f"现在\n获得 {current_comp_m} 元\n({year_str}{date_str}得0)"
                opt2_text = f"{year_str}{date_str}\n获得 {curr_amount} 元\n(现在得0)"
                standard_side = 'right'

            # 3. 运行选择
            trial_opt1.text = opt1_text
            trial_opt2.text = opt2_text

            trial_fixation.draw()
            win.flip()
            core.wait(0.2)  # 缩短注视点

            trial_opt1.draw()
            trial_opt2.draw()
            win.flip()

            kb_clock = core.Clock()
            keys = trial_key_resp.waitKeys(keyList=['left', 'right', 'escape'])
            rt = kb_clock.getTime()

            if 'escape' in keys:
                endExperiment(thisExp, win=win)
                return

            resp_key = keys[0]

            # 4. 快速回忆 (不等待太久)
            recall_text.text = "回忆数字，按空格查看"
            recall_text.draw()
            win.flip()
            event.waitKeys(keyList=['space'])

            reveal_text.text = mem_digits
            reveal_text.draw()
            # 缩短反馈流程
            visual.TextStim(win=win, text='按空格继续', pos=(0, -0.3), height=0.03).draw()
            win.flip()
            event.waitKeys(keyList=['space'])

            # 5. 数据记录
            thisExp.addData('block_amount', curr_amount)
            thisExp.addData('block_load', curr_load)
            thisExp.addData('block_delay', curr_delay)
            thisExp.addData('comp_m', current_comp_m)
            thisExp.addData('choice', resp_key)
            thisExp.addData('rt', rt)

            is_standard = (resp_key == standard_side)
            thisExp.addData('chose_delayed', is_standard)
            thisExp.nextEntry()

            # PEST 逻辑
            if prev_choice_is_standard is not None:
                if is_standard != prev_choice_is_standard:
                    pest_reversals += 1
                    current_step = max(current_step / 2, min_step_limit)

            if is_standard:
                current_comp_m += current_step
            else:
                current_comp_m -= current_step

            current_comp_m = int(max(1, current_comp_m))
            prev_choice_is_standard = is_standard

    # --- End Experiment ---
    end_text.draw()
    win.flip()
    core.wait(2)

    # 【关键】控制台打印保存路径，让用户放心
    print(f"\n======== 实验结束 ========")
    print(f"数据文件已保存至: {thisExp.dataFileName}.csv")
    print(f"==========================\n")

    endExperiment(thisExp, win=win)


if __name__ == '__main__':
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(expInfo=expInfo, thisExp=thisExp, win=win, globalClock='float')
    saveData(thisExp=thisExp)  # 确保显式调用保存
    quit(thisExp=thisExp, win=win)