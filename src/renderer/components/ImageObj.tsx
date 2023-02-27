import React from 'react';
import { useRecoilValue, useSetRecoilState } from 'recoil';
import { queueRunningState, initImagePathState } from '../atoms/atoms';
import { Image } from '@chakra-ui/react';
import Logo from '../images/ArtroomLogoTransparent.png';
import LoadingGif from '../images/loading.gif';

import ContextMenu from './ContextMenu/ContextMenu';
import ContextMenuItem from './ContextMenu/ContextMenuItem';
import ContextMenuList from './ContextMenu/ContextMenuList';
import ContextMenuTrigger from './ContextMenu/ContextMenuTrigger';
import { initImageState } from '../SettingsManager';

export default function ImageObj ({ B64, active } : { B64: string, active: boolean}) {
    const queueRunning = useRecoilValue(queueRunningState);
    const setInitImagePath = useSetRecoilState(initImagePathState);
    const setInitImage = useSetRecoilState(initImageState);

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
                    setInitImage(B64);
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
