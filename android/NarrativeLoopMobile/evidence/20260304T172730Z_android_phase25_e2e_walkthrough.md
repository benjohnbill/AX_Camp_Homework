# Android Phase2.5 E2E Walkthrough (20260304T172730Z)

- trace_id: trace-narrative_loop-20260305-rp25-it2
- task_id: T-nl-20260305-rp25-it2-android

## Device Window
`
List of devices attached
R3CR80HR90W            device product:c1qksw model:SM_N981N device:c1q transport_id:2
emulator-5554          device product:sdk_gphone64_x86_64 model:sdk_gphone64_x86_64 device:emu64xa transport_id:1


`

## SC-A (Plan-first complete on emulator)
- serial: emulator-5554
- mode: plan
- stage: 
- status: 
- evidence: 
- pass: False

## SC-B (Focus-first + retro complete on physical)
- serial: R3CR80HR90W
- mode: focus
- stage: 
- status: 
- evidence: 
- pass: False

## SC-C (OCR image_event -> reflect evidence_links)
- emulator image_event_id: 
- physical image_event_id: 
- emulator linked_count: -1
- physical linked_count: -1
- decision: False

## SC-D (physical/emulator same-window core flow)
- physical top activity snippet:
`
  * Task{9a1ef46 #2864 type=standard A=10620:com.example.narrativeloopmobile U=0 visible=true visibleRequested=true mode=fullscreen translucent=false sz=1}
    topResumedActivity=ActivityRecord{d964f21 u0 com.example.narrativeloopmobile/.MainActivity} t2864}
    * Hist  #0: ActivityRecord{d964f21 u0 com.example.narrativeloopmobile/.MainActivity} t2864}
      Intent { flg=0x10000000 cmp=com.example.narrativeloopmobile/.MainActivity }
      rootOfTask=true task=Task{9a1ef46 #2864 type=standard A=10620:com.example.narrativeloopmobile U=0 visible=true visibleRequested=true mode=fullscreen translucent=false sz=1}
      mActivityComponent=com.example.narrativeloopmobile/.MainActivity
  * Task{a4fc079 #1 type=home ?? U=0 visible=false visibleRequested=false mode=fullscreen translucent=true sz=1}
    * Task{53b5309 #2795 type=home I=com.microsoft.launcher/.Launcher U=0 rootTaskId=1 visible=false visibleRequested=false mode=fullscreen translucent=true sz=1}
        rootOfTask=true task=Task{53b5309 #2795 type=home I=com.microsoft.launcher/.Launcher U=0 rootTaskId=1 visible=false visibleRequested=false mode=fullscreen translucent=true sz=1}
  * Task{ce438fb #2796 type=recents ?? U=0 visible=false visibleRequested=false mode=fullscreen translucent=true sz=1}
    * Task{d879cc0 #2797 type=recents I=com.sec.android.app.launcher/com.android.quickstep.RecentsActivity U=0 rootTaskId=2796 visible=false visibleRequested=false mode=fullscreen translucent=true sz=1}
        rootOfTask=true task=Task{d879cc0 #2797 type=recents I=com.sec.android.app.launcher/com.android.quickstep.RecentsActivity U=0 rootTaskId=2796 visible=false visibleRequested=false mode=fullscreen translucent=true sz=1}
  * Task{aa1460f #2862 type=standard A=10132:com.samsung.android.messaging U=0 visible=false visibleRequested=false mode=fullscreen translucent=true sz=1}
      rootOfTask=true task=Task{aa1460f #2862 type=standard A=10132:com.samsung.android.messaging U=0 visible=false visibleRequested=false mode=fullscreen translucent=true sz=1}
`
- emulator top activity snippet:
`
  * Task{a1e1c80 #71 type=standard A=10227:com.example.narrativeloopmobile U=0 visible=true visibleRequested=true mode=fullscreen translucent=false sz=1}
    topResumedActivity=ActivityRecord{117597443 u0 com.example.narrativeloopmobile/.MainActivity t71}
    * Hist  #0: ActivityRecord{117597443 u0 com.example.narrativeloopmobile/.MainActivity t71}
      Intent { flg=0x10000000 xflg=0x5 cmp=com.example.narrativeloopmobile/.MainActivity }
      rootOfTask=true task=Task{a1e1c80 #71 type=standard A=10227:com.example.narrativeloopmobile}
      mActivityComponent=com.example.narrativeloopmobile/.MainActivity
  * Task{af14acd #1 type=home U=0 visible=false visibleRequested=false mode=fullscreen translucent=false sz=1}
    * Task{6e59586 #2 type=home I=com.google.android.apps.nexuslauncher/.NexusLauncherActivity U=0 rootTaskId=1 visible=false visibleRequested=false mode=fullscreen translucent=false sz=2}
        rootOfTask=true task=Task{6e59586 #2 type=home I=com.google.android.apps.nexuslauncher/.NexusLauncherActivity}
  * Task{737177d #48 name=SplitRoot type=undefined U=0 visible=false visibleRequested=false mode=fullscreen translucent=true sz=2}
    * Task{34b4672 #50 name=side type=undefined U=0 rootTaskId=48 visible=false visibleRequested=false mode=multi-window translucent=true sz=0}
    * Task{e408f40 #49 name=main type=undefined U=0 rootTaskId=48 visible=false visibleRequested=false mode=multi-window translucent=true sz=0}
    Resumed: ActivityRecord{117597443 u0 com.example.narrativeloopmobile/.MainActivity t71}
  ResumedActivity: ActivityRecord{117597443 u0 com.example.narrativeloopmobile/.MainActivity t71}
`
- decision: True

## Verdict
- success_gate: False
- logcat_path: D:\dev\Narrative_Loop\android\NarrativeLoopMobile\evidence\20260304T172730Z_android_phase25_e2e_logcat.log
