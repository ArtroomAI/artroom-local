import React from 'react';
import ArtroomIcon from '../images/icon.ico';
import MinimizeIcon from '../images/minimize.png';
import MaximizeIcon from '../images/maximize.png';
import CloseIcon from '../images/close.png';

const AppTopBar = () => {
    return (
        <div 
            id="menu-bar">
            <div className="left" role="menu">
                <img src={ArtroomIcon} width="30px"/>
                <h3 id="artroom-head">ArtroomAI</h3>
            </div>
            <div className="right">
                <button className="menubar-btn" id="minimize-btn" onClick={window.api.minimizeWindow}><img src={MinimizeIcon} width="20px"/></button>
                <button className="menubar-btn" id="max-unmax-btn" onClick={window.api.maxUnmaxWindow}><img src={MaximizeIcon} width="20px"/></button>
                <button className="menubar-btn" id="close-btn" onClick={window.api.closeWindow}><img src={CloseIcon} width="20px"/></button>
            </div>
        </div>
    );
}

export default AppTopBar;
