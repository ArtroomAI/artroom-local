import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import { Image } from '@chakra-ui/react';
import Logo from '../../images/ArtroomLogoTransparent.png';
import LoadingGif from '../../images/loading.gif';

import ContextMenu from '../ContextMenu/ContextMenu';
import ContextMenuItem from '../ContextMenu/ContextMenuItem';
import ContextMenuList from '../ContextMenu/ContextMenuList';
import ContextMenuTrigger from '../ContextMenu/ContextMenuTrigger';

export default function ImageObj ({ b64 = '', path = '', active } : { b64: string; path: string; active: boolean }) {
    const [imageSettings, setImageSettings] = useRecoilState(atom.imageSettingsState)

    const [queueRunning, setQueueRunning] = useRecoilState(atom.queueRunningState);
    const [initImagePath, setInitImagePath] = useRecoilState(atom.initImagePathState);

    const copyToClipboard = () => {
        window.api.copyToClipboard(b64);
    };

    const showInExplorer = () => {
        if (path !== '') {
            window.api.showInExplorer(path);
        }
    };

    return (
        <ContextMenu>
            <ContextMenuTrigger>
                {active
                    ? <Image
                        alignSelf="center"
                        fallbackSrc={queueRunning
                            ? LoadingGif
                            : Logo}
                        fit="scale-down"
                        h="55vh"
                        src={b64} />
                    : <Image
                        alignSelf="center"
                        fit="scale-down"
                        h="55vh"
                        src={b64} />}
            </ContextMenuTrigger>

            <ContextMenuList>
                <ContextMenuItem onClick={() => {
                    setInitImagePath('');
                    setImageSettings({...imageSettings, init_image: b64});
                } } colorScheme={undefined} disabled={false}>
                    Set As Starting Image
                </ContextMenuItem>

                <ContextMenuItem onClick={copyToClipboard} colorScheme={undefined} disabled={false}>
                    Copy To Clipboard
                </ContextMenuItem>

                <ContextMenuItem onClick={showInExplorer} colorScheme={undefined} disabled={false}>
                    Show In Explorer
                </ContextMenuItem>
            </ContextMenuList>
        </ContextMenu>
    );
}
