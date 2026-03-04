# Android Phase2.5 E2E Walkthrough (20260304T170804Z)

- trace_id: 	race-narrative_loop-20260305-rp25-it2
- task_id: T-nl-20260305-rp25-it2-android

## Device Window
`
List of devices attached
R3CR80HR90W            device product:c1qksw model:SM_N981N device:c1q transport_id:2
emulator-5554          device product:sdk_gphone64_x86_64 model:sdk_gphone64_x86_64 device:emu64xa transport_id:1


`

## SC-A (Plan-first complete on physical)
- serial: R3CR80HR90W
- mode: plan
- stage: 
- status: 
- evidence: 
- pass: False

## SC-B (Focus-first + retro complete on emulator)
- serial: emulator-5554
- mode: focus
- stage: 
- status: 
- evidence: 
- pass: False

## SC-C (OCR image_event -> reflect evidence_links)
- physical evidence linked: False
- emulator evidence linked: False
- decision: False

## SC-D (same-window dual-device core flow)
- physical top activity snippet:
`

  * Task{3f417e5 #2863 type=standard A=10620:com.example.narrativeloopmobile U=0 visible=true visibleRequested=true mode=fullscreen translucent=false sz=1}
    topResumedActivity=ActivityRecord{8314adc u0 com.example.narrativeloopmobile/.MainActivity} t2863}
    * Hist  #0: ActivityRecord{8314adc u0 com.example.narrativeloopmobile/.MainActivity} t2863}
      packageName=com.example.narrativeloopmobile processName=com.example.narrativeloopmobile
      app=ProcessRecord{6752d47 15398:com.example.narrativeloopmobile/u0a620}
      Intent { flg=0x10000000 cmp=com.example.narrativeloopmobile/.MainActivity }
      rootOfTask=true task=Task{3f417e5 #2863 type=standard A=10620:com.example.narrativeloopmobile U=0 visible=true visibleRequested=true mode=fullscreen translucent=false sz=1}
      taskAffinity=10620:com.example.narrativeloopmobile


`
- emulator top activity snippet:
`

  * Task{cbefb8c #62 type=standard A=10227:com.example.narrativeloopmobile U=0 visible=true visibleRequested=true mode=fullscreen translucent=false sz=1}
    topResumedActivity=ActivityRecord{82208959 u0 com.example.narrativeloopmobile/.MainActivity t62}
    * Hist  #0: ActivityRecord{82208959 u0 com.example.narrativeloopmobile/.MainActivity t62}
      packageName=com.example.narrativeloopmobile processName=com.example.narrativeloopmobile
      app=ProcessRecord{b264478 19436:com.example.narrativeloopmobile/u0a227}
      Intent { flg=0x10000000 xflg=0x5 cmp=com.example.narrativeloopmobile/.MainActivity }
      rootOfTask=true task=Task{cbefb8c #62 type=standard A=10227:com.example.narrativeloopmobile}
      taskAffinity=10227:com.example.narrativeloopmobile


`
- decision: False

## Notes
- Stream tab -> Run E2E Save path executed on both devices.
- Stage UI and action status are collected from on-device view hierarchy dump.
