async function getCkpts(data){
    return await window.api.getCkpts('getCkpts',data);
}

async function reinstallArtroom(){
    return await window.api.reinstallArtroom('reinstallArtroom');
}

async function getSettings(){
    return await window.api.getSettings('getSettings');
}

async function uploadSettings(){
    return await window.api.uploadSettings('uploadSettings');
}

async function chooseImages(){
    return await window.api.chooseImages('chooseImages');
}

async function getImageDir(){
    return await window.api.getImageDir('getImageDir');
}

async function openDiscord(){
    return await window.api.openDiscord('openDiscord');
}

async function openEquilibrium(){
    return await window.api.openEquilibrium('openEquilibrium');
}

async function uploadInitImage(){
    return await window.api.uploadInitImage('uploadInitImage');
}

async function getImageFromPath(data){
    return await window.api.getImageFromPath('getImageFromPath',data);
}


async function copyToClipboard(B64){
    return await window.api.copyToClipboard('copyToClipboard',B64);
}

async function chooseUploadPath(){
    return await window.api.chooseUploadPath('chooseUploadPath');
}

async function runPyTests(){
    return await window.api.runPyTests('runPyTests');
}

async function restartServer(isDebug){
    return await window.api.restartServer('restartServer', isDebug);
}