import React, { useState } from 'react';
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
    const [secondConfirmationNeeded, setSecondConfirmationNeeded] = useState(true);
    const { isOpen, onOpen, onClose } = useDisclosure();
    const cancelRef = React.useRef();

    const DebugInstaller = () => {
        window.api.pythonInstall();
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
                            Before Running: Try running Debug Mode by checking the Debug Mode option and pressing Save Settings.
                            </p>

                            <br />

                            <p>
                                Then, reach out on Discord to see if we can resolve this problem.
                            </p>
                            <br />
                            <p>
                                If all else fails, then proceed. NOTE: You will need a stable internet connection. This process could take a while so hang tight.  
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
                                        alert('This will uninstall and reinstall Artroom. Are you sure you want to continue?');
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
