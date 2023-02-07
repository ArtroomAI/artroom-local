import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import { Image } from '@chakra-ui/react';
import Logo from '../images/ArtroomLogoTransparent.png';
import LoadingGif from '../images/loading.gif';

import ContextMenu from './ContextMenu/ContextMenu';
import ContextMenuItem from './ContextMenu/ContextMenuItem';
import ContextMenuList from './ContextMenu/ContextMenuList';
import ContextMenuTrigger from './ContextMenu/ContextMenuTrigger';

export default function ImageObj ({ B64, active } : { B64: string, active: boolean}) {
    const [queueRunning, setQueueRunning] = useRecoilState(atom.queueRunningState);
    const [imageSettings, setImageSettings] = useRecoilState(atom.imageSettingsState)
    const [initImagePath, setInitImagePath] = useRecoilState(atom.initImagePathState);

    const copyToClipboard = () => {
        window.api.copyToClipboard(B64);
    };

    return (
        <ContextMenu>
            <ContextMenuTrigger>
                {active
                    ? <Image
                        fallbackSrc={queueRunning
                            ? LoadingGif
                            : Logo}
                        fit="scale-down"
                        h="55vh"
                        src={B64} />
                    : <Image
                        fit="scale-down"
                        h="55vh"
                        src={B64} />}
            </ContextMenuTrigger>

            <ContextMenuList>
                <ContextMenuItem onClick={() => {
                    setInitImagePath('');
                    setImageSettings({...imageSettings, init_image: B64});
                } } colorScheme={undefined} disabled={false}>
                    Set As Starting Image
                </ContextMenuItem>

                <ContextMenuItem onClick={() => {
                    copyToClipboard();
                } } colorScheme={undefined} disabled={false}>
                    Copy To Clipboard
                </ContextMenuItem>
            </ContextMenuList>
        </ContextMenu>
    );
}
