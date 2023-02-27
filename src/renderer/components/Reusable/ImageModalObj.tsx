import React from 'react';
import { useRecoilState, useSetRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import { Image } from '@chakra-ui/react';

import ContextMenu from './ContextMenu/ContextMenu';
import ContextMenuItem from './ContextMenu/ContextMenuItem';
import ContextMenuList from './ContextMenu/ContextMenuList';
import ContextMenuTrigger from './ContextMenu/ContextMenuTrigger';
import { initImageState } from '../../SettingsManager';

export default function ImageModalObj ({b64} : { b64: string }) {
    const setInitImagePath = useSetRecoilState(atom.initImagePathState);
    const setInitImage = useSetRecoilState(initImageState);

    const copyToClipboard = () => {
        window.api.copyToClipboard(b64);
    };

    return (
        <ContextMenu>
            <ContextMenuTrigger>
                 <Image
                    maxHeight="600px"
                    maxWidth="600px"
                    borderRadius="5"
                    src={b64} />
            </ContextMenuTrigger>

            <ContextMenuList>
                <ContextMenuItem disabled = {false} colorScheme={'white'} onClick={() => {
                    setInitImagePath('');
                    setInitImage(b64);
                }}>
                    Set As Starting Image
                </ContextMenuItem>

                <ContextMenuItem disabled = {false} colorScheme={'white'} onClick={() => {
                    copyToClipboard();
                }}>
                    Copy To Clipboard
                </ContextMenuItem>
            </ContextMenuList>
        </ContextMenu>
    );
}
