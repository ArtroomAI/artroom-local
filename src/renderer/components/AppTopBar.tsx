import React from 'react'
import ArtroomIcon from '../images/artroom_icon.ico'
import MinimizeIcon from '../images/minimize.png'
import MaximizeIcon from '../images/maximize.png'
import CloseIcon from '../images/close.png'
import { useRecoilValue } from 'recoil'
import { cloudModeState } from '../atoms/atoms.login'
import { connectedToServerState } from '../SettingsManager'

const ConnectionStatus = () => {
  const isConnectedToServer = useRecoilValue(connectedToServerState)
  return (
    <div
      style={{
        marginLeft: '5px',
        width: '14px',
        height: '14px',
        borderRadius: '50%',
        backgroundColor: isConnectedToServer ? '#2f2' : '#f22',
      }}
    />
  )
}

const AppTopBar = () => {
  const isCloudMode = useRecoilValue(cloudModeState)

  return (
    <div id="menu-bar">
      <div className="left" role="menu">
        <img src={ArtroomIcon} width="30px" />
        <h3 id="artroom-head">ArtroomAI{isCloudMode ? ' Cloud' : ''}</h3>
        <ConnectionStatus />
      </div>
      <div className="right">
        <button className="menubar-btn" id="minimize-btn" onClick={window.api.minimizeWindow}>
          <img src={MinimizeIcon} width="20px" />
        </button>
        <button className="menubar-btn" id="max-unmax-btn" onClick={window.api.maxUnmaxWindow}>
          <img src={MaximizeIcon} width="20px" />
        </button>
        <button className="menubar-btn" id="close-btn" onClick={window.api.closeWindow}>
          <img src={CloseIcon} width="20px" />
        </button>
      </div>
    </div>
  )
}

export default AppTopBar
