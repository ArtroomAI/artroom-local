import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import { Image } from '@chakra-ui/react';
import Logo from '../../images/ArtroomLogoTransparent.png';
import LoadingGif from '../../images/loading.gif';

import ContextMenu from './ContextMenu/ContextMenu';
import ContextMenuItem from './ContextMenu/ContextMenuItem';
import ContextMenuList from './ContextMenu/ContextMenuList';
import ContextMenuTrigger from './ContextMenu/ContextMenuTrigger';

export default function ImageModalObj ({b64}) {
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);
    const [initImagePath, setInitImagePath] = useRecoilState(atom.initImagePathState);

    const copyToClipboard = () => {
        window.copyToClipboard(b64);
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
            </ContextMenuList>
        </ContextMenu>
    );
}
