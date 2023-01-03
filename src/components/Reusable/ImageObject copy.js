import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import { Image,  Modal, Button, Stack, Box} from '@chakra-ui/react';

import ContextMenu from './ContextMenu/ContextMenu';
import ContextMenuItem from './ContextMenu/ContextMenuItem';
import ContextMenuList from './ContextMenu/ContextMenuList';
import ContextMenuTrigger from './ContextMenu/ContextMenuTrigger';

export default function ImageObject ({b64, metadata, isOpen, onOpen, onClose}) {
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);
    const [initImagePath, setInitImagePath] = useRecoilState(atom.initImagePathState);

    const copyToClipboard = () => {
        window.copyToClipboard(b64);
    };

    return (
        <>
        <Modal isOpen={isOpen} onClose={onClose}>
            <Stack>
            <Image src={b64} />
            <Box>
                <p>{metadata}</p>
            </Box>
            <Button onClick={() => copyToClipboard("settings")}>Copy Settings</Button>
            <Button onClick={() => copyToClipboard("image")}>Copy Image</Button>
            <Button onClick={() => share("image")}>Share Image</Button>
            </Stack>
        </Modal>
        <ContextMenu>
            <ContextMenuTrigger>
                <Image
                    src={b64} 
                    borderRadius="3%"
                    onClick={onOpen}
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
        </>
    );
}
