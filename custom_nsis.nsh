!include UAC.nsh
!include MUI.nsh
!include nsDialogs.nsh
!include LogicLib.nsh


!macro RunInstaller
  SetDetailsView show
  DetailPrint "Installing Model Weights and CONDA dependencies"
  ${if} ${isUpdated}
    ;${StdUtils.ExecShellWaitEx} $0 $1 "$INSTDIR\py_cuda_install.exe" "open" "1"
  ${Else}
    ${StdUtils.ExecShellWaitEx} $0 $1 "$INSTDIR\py_cuda_install.exe" "open" "$skipWeights $logDir"
  ${endIf}

  DetailPrint "Result: $0 -> $1" ;returns "ok", "no_wait" or "error".
	StrCmp $0 "error" ExecFailed ;check if process failed to create
	StrCmp $0 "no_wait" WaitNotPossible ;check if process can be waited for - always check this!
	StrCmp $0 "ok" WaitForProc ;make sure process was created successfully
	Abort
	
	WaitForProc:
	DetailPrint "Waiting for process. ZZZzzzZZZzzz..."
	${StdUtils.WaitForProcEx} $2 $1
  ${If} $2 == 0
    MessageBox MB_OK "Install Successfull"
    Goto WaitDone
  ; ${Else}
  ;   MessageBox MB_OK "FAILURE: Please let the CMD terminal finish installing. Uninstall and try reinstall again (exit code: $2)" 
  ${EndIf}
	
	ExecFailed:
	MessageBox MB_OK "FAILURE: Please let the CMD terminal finish installing. Uninstall and try reinstall again (exit code: $2)" 
	Goto WaitDone

	WaitNotPossible:
	MessageBox MB_OK "Can not wait for process."
	Goto WaitDone
	WaitDone:
!macroend



!macro customInstall
  ;!insertmacro customDirectory
  MessageBox MB_OK "Upon Pressing OK, the installer will install Conda dependencies and model weights to $logDir . Please wait until the installer is finished installing."
  !insertmacro RunInstaller
!macroend

; !macro getArtroomDir
;   ;!insertmacro customDirectory
;   Call pageGetLogDir
;   Pop $0
;   MessageBox MB_OK "result: $0"
; !macroend


;log dir variable and its default 
Var /global logDir
Var /global skipWeights
;var /global defaultLogDir 
!define defaultLogDir "$PROFILE"

    ; finally, use that custom page in between your other pages:
Page custom pageGetLogDir

;three functions for the custom page
Function pageGetLogDir
  !insertmacro MUI_HEADER_TEXT "Choose artroom folder Destination" "This folder will contain Python dependencies and model weights (~17GB)"

  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}

  var /global browseButton
  var /global directoryText
  var /global chkbx

  StrCpy $logDir "${defaultLogDir}"
  StrCpy $skipWeights "0"

  ${NSD_CreateLabel} 0 0 100% 36u "Artroom folder will be written to the following path. To have them written to a different path, click Browse and select another folder. Click Next to continue."
  Pop $0 ; ignore

  ${NSD_CreateText} 0 37u 75% 12u "$logDir"
  pop $directoryText
  ${NSD_OnChange} $directoryText onLogDirTextChange

  ;create button, save handle and connect to function
  ${NSD_CreateBrowseButton} 80% 36u 20% 14u "Browse..."
  pop $browseButton
  ${NSD_OnClick} $browseButton onLogDirButtonClick

  ${NSD_CreateCheckbox} 0 60u 100% 30% "Skip Installation of Model Weights $\n (ONLY check this option if you already have model weights installed)"
  Pop $chkbx
  ${NSD_OnClick} $chkbx onCheckBoxChange

  ;${NSD_Return} $directoryText

  nsDialogs::Show
FunctionEnd

Function onLogDirButtonClick
  nsDialogs::SelectFolderDialog "Select Log File Destination" ""
  Pop $0
  ${If} $0 != error
    StrCpy $logDir $0
    ${NSD_SetText} $directoryText $logDir
  ${EndIf}
FunctionEnd

Function onLogDirTextChange
  ${NSD_GetText} $directoryText $logDir
FunctionEnd

Function onCheckBoxChange
  Pop $chkbx
  ${NSD_GetState} $chkbx $0
	StrCpy $skipWeights $0
FunctionEnd


; !macro customUninstall
;   ${ifNot} ${isUpdated}
;     RMDir /r "$PROFILE\artroom"
;   ${endIf}
; !macroend



  ; !macro RunInstaller
;   ${StdUtils.ExecShellWaitEx} $0 $1 "$INSTDIR\py_cuda_install.exe" "open" ""
;   MessageBox MB_OK "Test message1: $0"
;   MessageBox MB_OK "Test message2: $1"
; !macroend
  
; !addplugindir $INSTDIR\nsis\Plugins
; !macro RunInstaller
;   nsExec::ExecToStack '"$0" $INSTDIR\\py_cuda_install.exe'
;   Pop $0 # return value/error/timeout
;   Pop $1 # printed text, up to ${NSIS_MAX_STRLEN}
;   DetailPrint '"${NSISDIR}\makensis.exe" /VERSION printed: $1'
;   DetailPrint ""
;   DetailPrint "       Return value: $0"
;   DetailPrint ""
;   MessageBox MB_OK "Test message: $0"
; !macroend


; !macro RunInstaller
;   SetDetailsView show
;   ExecDos::exec /NOUNLOAD /ASYNC /DETAILED "$INSTDIR\py_cuda_install.exe"
;   Pop $0 # thread handle for wait
;   DetailPrint "$0"
;   # you can add some installation code here to execute while application is running.
;   ExecDos::wait $0
;   Pop $0 # return value
;   MessageBox MB_OK "Exit code $0"
; !macroend
