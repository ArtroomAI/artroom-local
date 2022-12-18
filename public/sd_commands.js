async function getCkpts (data) {
    await window.api.getCkpts(
        'getCkpts',
        data
    );
}

async function reinstallArtroom () {
    await window.api.reinstallArtroom('reinstallArtroom');
}

async function getSettings () {
    await window.api.getSettings('getSettings');
}

async function uploadSettings () {
    await window.api.uploadSettings('uploadSettings');
}

async function chooseImages () {
    await window.api.chooseImages('chooseImages');
}

async function getImageDir () {
    await window.api.getImageDir('getImageDir');
}

async function openDiscord () {
    await window.api.openDiscord('openDiscord');
}

async function openEquilibrium () {
    await window.api.openEquilibrium('openEquilibrium');
}

async function uploadInitImage () {
    await window.api.uploadInitImage('uploadInitImage');
}

async function getImageFromPath (data) {
    await window.api.getImageFromPath(
        'getImageFromPath',
        data
    );
}


async function copyToClipboard (B64) {
    await window.api.copyToClipboard(
        'copyToClipboard',
        B64
    );
}

async function chooseUploadPath () {
    await window.api.chooseUploadPath('chooseUploadPath');
}

async function runPyTests () {
    await window.api.runPyTests('runPyTests');
}

async function restartServer (isDebug) {
    await window.api.restartServer(
        'restartServer',
        isDebug
    );
}

async function mergeModels (data) {
    await window.api.mergeModels(
        'mergeModels',
        data
    );
}
