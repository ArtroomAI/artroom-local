import React, { useState } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import axios from 'axios';
import {
    useDisclosure,
    AlertDialog,
    AlertDialogOverlay,
    AlertDialogContent,
    AlertDialogHeader,
    AlertDialogBody,
    AlertDialogFooter,
    Button
} from '@chakra-ui/react';

function DebugInstallerModal () {
    const [secondConfirmationNeeded, setSecondConfirmationNeeded] = useState(false);
    const { isOpen, onOpen, onClose } = useDisclosure();
    const cancelRef = React.useRef();

    const DebugInstaller = (event) => {
        window.reinstallArtroom();
        onClose();
    };

    return (
        <>
            <Button
                backgroundColor="red.600"
                colorScheme="red"
                onClick={onOpen}>
                Debug Artroom Install
            </Button>

            <AlertDialog
                isOpen={isOpen}
                leastDestructiveRef={cancelRef}
                onClose={onClose}
                returnFocusOnClose
            >
                <AlertDialogOverlay>
                    <AlertDialogContent bg="gray.800">
                        <AlertDialogHeader
                            fontSize="lg"
                            fontWeight="bold">
                            Debug Artroom Install
                        </AlertDialogHeader>

                        <AlertDialogBody>
                            <p>
                                WARNING: This will reinstall Artroom. ONLY DO THIS IF YOU HAVE A BROKEN INSTALL. A command prompt will open and you will see the installer again, except this one will stay open until you press a button.
                            </p>

                            <br />

                            <p>
                                BEFORE YOU RUN THIS!!!
                            </p>

                            <br />

                            <p>
                                Try running Debug Mode by checking the Debug Mode option and pressing Save Settings.
                            </p>

                            <br />

                            <p>
                                Then, reach out on Discord to see if we can resolve this problem.
                            </p>

                            <br />

                            <p>
                                IF ALL ELSE FAILS, then go ahead and hopefully this will fix it.
                            </p>

                        </AlertDialogBody>

                        <AlertDialogFooter>
                            <Button
                                onClick={onClose}
                                ref={cancelRef}>
                                Cancel
                            </Button>

                            <Button
                                backgroundColor="red.600"
                                colorScheme="red"
                                ml={3}
                                onClick={() => {
                                    if (!secondConfirmationNeeded) {
                                        setSecondConfirmationNeeded(true);
                                        alert('This will uninstall and reinstall Artroom. If you have model weights you want to keep that are installed in artroom/model_weights, please move them to a separate folder before moving on.  Are you sure you want to continue?');
                                    } else {
                                        setSecondConfirmationNeeded(false);
                                        DebugInstaller();
                                    }
                                }}>
                                Reinstall Artroom Backend
                            </Button>
                        </AlertDialogFooter>
                    </AlertDialogContent>
                </AlertDialogOverlay>
            </AlertDialog>
        </>
    );
}

export default DebugInstallerModal;
