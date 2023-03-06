import React, { useCallback, useState } from 'react';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import { Image, Modal, ModalBody, ModalContent, ModalOverlay } from '@chakra-ui/react';
import Logo from '../../images/ArtroomLogoTransparent.png';
import LoadingGif from '../../images/loading.gif';

import ContextMenu from '../ContextMenu/ContextMenu';
import ContextMenuItem from '../ContextMenu/ContextMenuItem';
import ContextMenuList from '../ContextMenu/ContextMenuList';
import ContextMenuTrigger from '../ContextMenu/ContextMenuTrigger';
import { initImageState } from '../../SettingsManager';

export default function ImageObj ({ b64 = '', path = '', active } : { b64: string; path: string; active: boolean }) {
    const queueRunning = useRecoilValue(atom.queueRunningState);
    const setInitImagePath = useSetRecoilState(atom.initImagePathState);

    const setInitImage = useSetRecoilState(initImageState);

    const [showModal, setShowModal] = useState(false);
    
    const handleClose = useCallback(() => {
        setShowModal(false);
    }, []);
    const handleOpen = useCallback(() => {
        if(b64) {
            setShowModal(true);
        }
    }, [b64]);

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
                        src={b64}
                        onClick={handleOpen} />
                    : <Image
                        alignSelf="center"
                        fit="scale-down"
                        h="55vh"
                        src={b64}
                        onClick={handleOpen} />}
                <Modal
                    size='6xl'
                    isOpen={showModal}
                    onClose={handleClose}
                    scrollBehavior='outside'>
                        <ModalOverlay bg='blackAlpha.900' />
                        <ModalContent>
                            <ModalBody display="flex" justifyContent="center">
                                <Image
                                    alignSelf="center"
                                    fit="scale-down"
                                    h="100%"
                                    src={b64}/>
                            </ModalBody>
                        </ModalContent>
                </Modal>
            </ContextMenuTrigger>

            <ContextMenuList>
                <ContextMenuItem onClick={() => {
                    setInitImage(b64);
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