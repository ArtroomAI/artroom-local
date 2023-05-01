import { clipboardHandles } from "./handles/clipboardHandles";
import { setupImageViewer } from "./handles/imageViewer";
import { installerHandles } from "./handles/installerHandles";
import { setupQueueHandles } from "./handles/queueHandles";
import { exposeMenuFunctions } from "./handles/menu-functions";
import { openLinksHandles } from "./handles/openLinksHandles";
import { settingsLoading } from "./handles/settingsLoading";
import { filesHandles } from "./handles/filesHandles";
import { pytestHandles } from "./handles/pytestHandles";

export const setupHandles = (browserWindow: Electron.BrowserWindow) => {
    exposeMenuFunctions(browserWindow);
    installerHandles(browserWindow);
    setupImageViewer(browserWindow);
    settingsLoading();
    setupQueueHandles();
    clipboardHandles();
    openLinksHandles();
    filesHandles(browserWindow);
    pytestHandles();
}
