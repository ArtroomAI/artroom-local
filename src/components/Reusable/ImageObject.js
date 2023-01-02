import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import { Image } from '@chakra-ui/react';

import ContextMenu from './ContextMenu/ContextMenu';
import ContextMenuItem from './ContextMenu/ContextMenuItem';
import ContextMenuList from './ContextMenu/ContextMenuList';
import ContextMenuTrigger from './ContextMenu/ContextMenuTrigger';

export default function ImageObject ({b64, metadata}) {
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);
    const [initImagePath, setInitImagePath] = useRecoilState(atom.initImagePathState);

    const copyToClipboard = () => {
        window.copyToClipboard(b64);
    };

    return (
        <ContextMenu>
            <ContextMenuTrigger>
                <Image
                    src={b64} 
                    borderRadius="3%"
                />
            </ContextMenuTrigger>
            <ContextMenuList>
                <ContextMenuItem onClick={() => {
                    setInitImagePath('');
                    setInitImage(b64);
                }}>
                    Set As Starting Image
                </ContextMenuItem>
                <ContextMenuItem onClick={() => {
                    copyToClipboard();
                }}>
                    Copy To Clipboard
                </ContextMenuItem>
                {metadata &&                 
                    <ContextMenuItem onClick={() => {
                        copyToClipboard();
                    }}>
                        {metadata}
                    </ContextMenuItem>
                }
            </ContextMenuList>
        </ContextMenu>
    );
}
