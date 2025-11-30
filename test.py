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

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'loop'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='D:\\Study\\2025-2026-1\\NeuralScience\\project\\test\\loop.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('intro_key_resp') is None:
        # initialise intro_key_resp
        intro_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='intro_key_resp',
        )
    if deviceManager.getDevice('trial_key_resp') is None:
        # initialise trial_key_resp
        trial_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='trial_key_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Intro" ---
    intro_text = visual.TextStim(win=win, name='intro_text',
        text='欢迎参与实验！\n每个试次会呈现两个选项，例如：\n左侧："立即获得50元"  右侧："7天后获得100元"\n请按【左箭头】选择左侧选\n按【右箭头】选择右侧选项\n选择你更偏好的选项即可\n按"空格"开始实验',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_key_resp = keyboard.Keyboard(deviceName='intro_key_resp')
    
    # --- Initialize components for Routine "trial" ---
    trial_fixation = visual.TextStim(win=win, name='trial_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # 左选项文本（位置：左半屏）
    trial_opt1 = visual.TextStim(win=win, name='trial_opt1',
        text='',  # 内容后续动态赋值
        font='Arial',
        pos=(-0.3, 0),  # 左移0.3单位（避免与右选项重叠）
        draggable=False, height=0.035, wrapWidth=0.6, ori=0.0,  # 限制宽度，避免换行混乱
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    # 右选项文本（位置：右半屏）
    trial_opt2 = visual.TextStim(win=win, name='trial_opt2',
        text='',  # 内容后续动态赋值
        font='Arial',
        pos=(0.3, 0),  # 右移0.3单位
        draggable=False, height=0.035, wrapWidth=0.6, ori=0.0,
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);

    trial_key_resp = keyboard.Keyboard(deviceName='trial_key_resp')
    
    # --- Initialize components for Routine "end" ---
    end_text = visual.TextStim(win=win, name='end_text',
        text='实验结束！感谢你的参与！\n3秒后自动退出',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);

    # --- 【替换】为以下代码 ---

    # 1. 设置实验开始的“今天”
    experiment_start_date = datetime.now()

    # 2. 升级后的延迟时间列表 (对数分布) - 使用小金额组作为默认
    # 格式：(延迟天数, 固定金额)
    standard_options = [
        (1, 100),  # 1天
        (7, 100),  # 1周
        (30, 100),  # 1个月
        (90, 100),  # 3个月
        (180, 100),  # 半年
        (365, 100),  # 1年
        (3650, 100)  # 10年 (捕捉极度耐心)
    ]
    # 随机打乱区块顺序
    shuffle(standard_options)

    # 2. PEST算法参数（保持不变，微调步长）
    pest_params = {
        'initial_comp_m': 50,
        'initial_step': 25,  # 建议改为25
        'min_step': 1,  # 建议改为1
        'max_reversals': 6
    }

    # 3. 固定比较延迟
    fixed_comp_t = 0

    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Intro" ---
    # create an object to store info about Routine Intro
    Intro = data.Routine(
        name='Intro',
        components=[intro_text, intro_key_resp],
    )
    Intro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for intro_key_resp
    intro_key_resp.keys = []
    intro_key_resp.rt = []
    _intro_key_resp_allKeys = []
    # store start times for Intro
    Intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Intro.tStart = globalClock.getTime(format='float')
    Intro.status = STARTED
    thisExp.addData('Intro.started', Intro.tStart)
    Intro.maxDuration = None
    # keep track of which components have finished
    IntroComponents = Intro.components
    for thisComponent in Intro.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Intro" ---
    Intro.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_text* updates
        
        # if intro_text is starting this frame...
        if intro_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_text.frameNStart = frameN  # exact frame index
            intro_text.tStart = t  # local t and not account for scr refresh
            intro_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_text.started')
            # update status
            intro_text.status = STARTED
            intro_text.setAutoDraw(True)
        
        # if intro_text is active this frame...
        if intro_text.status == STARTED:
            # update params
            pass
        
        # *intro_key_resp* updates
        waitOnFlip = False
        
        # if intro_key_resp is starting this frame...
        if intro_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_key_resp.frameNStart = frameN  # exact frame index
            intro_key_resp.tStart = t  # local t and not account for scr refresh
            intro_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_key_resp.started')
            # update status
            intro_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = intro_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_key_resp_allKeys.extend(theseKeys)
            if len(_intro_key_resp_allKeys):
                intro_key_resp.keys = _intro_key_resp_allKeys[-1].name  # just the last key pressed
                intro_key_resp.rt = _intro_key_resp_allKeys[-1].rt
                intro_key_resp.duration = _intro_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Intro,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Intro.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Intro.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Intro" ---
    for thisComponent in Intro.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Intro
    Intro.tStop = globalClock.getTime(format='float')
    Intro.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Intro.stopped', Intro.tStop)
    # check responses
    if intro_key_resp.keys in ['', [], None]:  # No response was made
        intro_key_resp.keys = None
    thisExp.addData('intro_key_resp.keys',intro_key_resp.keys)
    if intro_key_resp.keys != None:  # we had a response
        thisExp.addData('intro_key_resp.rt', intro_key_resp.rt)
        thisExp.addData('intro_key_resp.duration', intro_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "Intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    # --- 【替换原有TrialHandler2】区块循环+PEST动态试次 ---
    # 遍历每个区块（每个区块对应一个标准延迟）
    for (standard_t, standard_m) in standard_options:
        # 【新增】区块间休息提示（防止疲劳，每1个区块休息5秒）
        rest_text = visual.TextStim(win=win, text=f'准备开始下一组选择\n按"空格"继续（之后有5秒休息）', 
                                    pos=(0,0), height=0.04, color='white')
        rest_text.draw()
        win.flip()
        # 等待被试按空格
        rest_key_resp = keyboard.Keyboard()
        rest_key_resp.waitKeys(keyList=['space', 'escape'])
        if 'escape' in rest_key_resp.keys:
            endExperiment(thisExp, win=win)
        # 休息5秒
        rest_text.text = '休息5秒...'
        rest_text.draw()
        win.flip()
        core.wait(5)

        # 【新增】休息结束后清空窗口，避免残留文本干扰后续试次
        win.clearBuffer()
        win.flip()

        # --- 初始化当前区块的PEST变量 ---
        current_comp_m = pest_params['initial_comp_m']  # 当前比较金额（动态调整）
        current_step = pest_params['initial_step']      # 当前调整步长
        reversal_count = 0                              # 反转次数（判断是否结束区块）
        prev_choice_is_standard = None                  # 上一次是否选了标准选项（判断反转）

        # --- 当前区块的动态试次循环 ---
        while reversal_count < pest_params['max_reversals']:

            # --- 【删除】原来的 if random() < 0.5: ... else ... 这一整块逻辑 ---

            # --- 【插入】下面这一大段新代码 ---

            # A. 计算未来的具体日期
            # standard_t 是当前区块的延迟天数
            future_date = experiment_start_date + timedelta(days=standard_t)

            # B. 格式化日期字符串
            # Windows系统下，strftime如果不支持中文，可改用 f"{future_date.month}月{future_date.day}日"
            try:
                date_str = future_date.strftime("%m月%d日")
                year_str = future_date.strftime("%Y年")
            except:
                # 备用方案，防止某些系统中文编码报错
                date_str = f"{future_date.month}月{future_date.day}日"
                year_str = f"{future_date.year}年"

            # C. 生成选项文本 (结合 Date Framing 和 Explicit Zero)
            if random() < 0.5:
                # 情况1：左=标准选项（延迟），右=比较选项（即时）
                # 文本示例："2025年5月28日\n获得 100 元\n(现在获得 0 元)"
                opt1_content = f"{year_str}{date_str}\n获得 {standard_m} 元\n(现在获得 0 元)"

                # 文本示例："现在\n获得 50 元\n(2025年5月28日获得 0 元)"
                opt2_content = f"现在\n获得 {current_comp_m} 元\n({year_str}{date_str}获得 0 元)"

                standard_side = 'left'
            else:
                # 情况2：左右交换
                opt1_content = f"现在\n获得 {current_comp_m} 元\n({year_str}{date_str}获得 0 元)"
                opt2_content = f"{year_str}{date_str}\n获得 {standard_m} 元\n(现在获得 0 元)"
                standard_side = 'right'

            # --- 【插入结束】 ---

            # --- 2. 运行单个试次的Routine（注视点→选项呈现→反应） ---
            # 准备试次Routine（沿用原有Routine结构）
            trial = data.Routine(
                name='trial',
                components=[trial_fixation, trial_opt1, trial_opt2, trial_key_resp],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # 初始化反应变量
            trial_key_resp.keys = []
            trial_key_resp.rt = []
            _trial_key_resp_allKeys = []
            # 重置计时器
            trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial.tStart = globalClock.getTime(format='float')
            trial.status = STARTED
            thisExp.addData('trial.started', trial.tStart)
            for thisComponent in trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.status = NOT_STARTED
            t = 0
            frameN = -1

            # --- 单个试次的帧循环（注视点200ms→选项呈现→反应） ---
            while continueRoutine:
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                frameN += 1

                # ① 呈现注视点（前200ms）
                if trial_fixation.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                    trial_fixation.setAutoDraw(True)
                    trial_fixation.status = STARTED
                # 200ms后隐藏注视点
                if trial_fixation.status == STARTED and tThisFlip >= 0.2 - frameTolerance:
                    trial_fixation.setAutoDraw(False)
                    trial_fixation.status = FINISHED

                # ② 呈现两个选项（注视点消失后，即t≥0.2s时）
                if trial_opt1.status == NOT_STARTED and tThisFlip >= 0.2 - frameTolerance:
                    # 设置左右选项内容
                    trial_opt1.text = opt1_content
                    trial_opt2.text = opt2_content
                    trial_opt1.setAutoDraw(True)
                    trial_opt2.setAutoDraw(True)
                    trial_opt1.status = STARTED
                    trial_opt2.status = STARTED
                    # 同时启动反应按键监听
                    trial_key_resp.status = STARTED
                    win.callOnFlip(trial_key_resp.clock.reset)
                    win.callOnFlip(trial_key_resp.clearEvents)

                # ③ 监听反应（左/右箭头）
                if trial_key_resp.status == STARTED:
                    theseKeys = trial_key_resp.getKeys(keyList=['left', 'right', 'escape'], ignoreKeys=["escape"])
                    _trial_key_resp_allKeys.extend(theseKeys)
                    if len(_trial_key_resp_allKeys):
                        trial_key_resp.keys = _trial_key_resp_allKeys[-1].name
                        trial_key_resp.rt = _trial_key_resp_allKeys[-1].rt
                        # 反应后结束试次
                        continueRoutine = False

                # 检查退出（Esc键）
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    endExperiment(thisExp, win=win)
                if thisExp.status == PAUSED:
                    pauseExperiment(thisExp, win=win, timers=[routineTimer, globalClock], currentRoutine=trial)

                # 刷新屏幕
                if continueRoutine:
                    win.flip()

            # --- 单个试次结束：隐藏刺激+记录数据 ---
            # 隐藏所有刺激
            for thisComponent in trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # 记录试次数据（关键！方便后续拟合）
            thisExp.addData('standard_t_days', standard_t)    # 标准延迟（天）
            thisExp.addData('standard_m_yuan', standard_m)    # 标准金额（元）
            thisExp.addData('comp_t_days', fixed_comp_t)      # 比较延迟（天）
            thisExp.addData('comp_m_yuan', current_comp_m)    # 比较金额（元）
            thisExp.addData('opt1_content', opt1_content)     # 左选项内容
            thisExp.addData('opt2_content', opt2_content)     # 右选项内容
            thisExp.addData('choice_key', trial_key_resp.keys)# 选择的按键（left/right）
            thisExp.addData('rt', trial_key_resp.rt)          # 反应时（秒）
            thisExp.addData('reversal_count', reversal_count) # 当前区块反转次数
            # 判断是否选择了标准选项（用于PEST调整）
            current_choice_is_standard = (trial_key_resp.keys == standard_side)
            thisExp.addData('is_choose_standard', current_choice_is_standard)  # 是否选标准选项
            thisExp.nextEntry()  # 保存当前试次数据
            routineTimer.reset()

            # --- PEST算法：调整下一个比较金额+判断反转 ---
            if prev_choice_is_standard is not None:  # 跳过第一次试次（无历史选择）
                # 出现反转（上一次选标准，这次选比较；或反之）
                if current_choice_is_standard != prev_choice_is_standard:
                    reversal_count += 1
                    current_step = max(current_step / 2, pest_params['min_step'])  # 步长减半（不小于最小步长）

            # 调整下一个比较金额
            if current_choice_is_standard:
                # 选了标准选项（说明当前比较金额太低）→ 增加比较金额
                current_comp_m += current_step
            else:
                # 选了比较选项（说明当前比较金额太高）→ 减少比较金额
                current_comp_m -= current_step
            # 确保金额不为负（避免逻辑错误）
            current_comp_m = max(1, current_comp_m)

            # 更新上一次的选择（为下一次判断反转做准备）
            prev_choice_is_standard = current_choice_is_standard

    # --- 所有区块结束：运行end Routine ---
    # 呈现结束文本
    end = data.Routine(name='end', components=[end_text])
    end.status = NOT_STARTED
    continueRoutine = True
    end.tStart = globalClock.getTime(format='float')
    end_text.setAutoDraw(True)
    win.flip()
    core.wait(3)  # 显示3秒后退出
    end_text.setAutoDraw(False)

    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    thisExp.nextEntry()
    # the Routine "end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
