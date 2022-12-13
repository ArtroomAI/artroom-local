async function startSD(data){
    return await window.api.startSD('startSD',data);
}

async function paintSD(data){
    return await window.api.paintSD('paintSD',data);
}

async function updateConfig(data){
    return await window.api.updateConfig('updateConfig',data);
}

async function updateSettings(data){
    return await window.api.updateSettings('updateSettings',data);
}

async function uploadInitImage(){
    return await window.api.uploadInitImage('uploadInitImage');
}
async function upscale(data){
    return await window.api.upscale('upscale',data);
}

async function getImage(){
    return await window.api.getImage('getImage');
}

async function getImageFromPath(data){
    return await window.api.getImageFromPath('getImageFromPath',data);
}

async function getModelCkpts(data){
    return await window.api.getModelCkpts('getModelCkpts',data);
}

async function getImages(data){
    return await window.api.getImages('getImages',data);
}

async function updateSettings(data){
    return await window.api.updateSettings('updateSettings',data);
}

async function chooseImages(){
    return await window.api.chooseImages('chooseImages');
}

async function getSettings(){
    return await window.api.getSettings('getSettings');
}

async function getQueue(){
    return await window.api.getQueue('getQueue');
}

async function writeQueue(data){
    return await window.api.updateSettings('writeQueue',data);
}

async function getUpscaleSettings(){
    return await window.api.getUpscaleSettings('getUpscaleSettings');
}


async function getImageDir(){
    return await window.api.getImageDir('getImageDir');
}

async function chooseUploadPath(){
    return await window.api.chooseUploadPath('chooseUploadPath');
}

async function uploadSettings(){
    return await window.api.uploadSettings('uploadSettings');
}