import React, { useState } from 'react';
import {
    useDisclosure,
    AlertDialog,
    AlertDialogOverlay,
    AlertDialogContent,
    AlertDialogHeader,
    AlertDialogBody,
    AlertDialogFooter,
    Button,
    Text,
    Flex,
    Switch,
    useToast
} from '@chakra-ui/react';

function DebugInstallerModal () {
    const { isOpen, onOpen, onClose } = useDisclosure();
    const toast = useToast({});

    const cancelRef = React.useRef();
    const [useAMDInstaller, setUseAMDInstaller] = useState(false);
    const DebugInstaller = () => {
        window.api.pythonInstall(useAMDInstaller);
        toast({
            title: 'Reinstalling Artroom Backend',
            status: 'success',
            position: 'top',
            duration: 2000,
            isClosable: true,
            containerStyle: {
                pointerEvents: 'none'
            }
        });
        onClose();
    };

    return (
        <>
            <Button
                backgroundColor="red.600"
                colorScheme="red"
                onClick={onOpen}>
                Reinstall Artroom Backend
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
                            <Text fontSize="lg" fontWeight="medium">
                                WARNING:
                            </Text>
                            <p>
                                This will reinstall Artroom. ONLY DO THIS IF YOU HAVE A BROKEN INSTALL. A command prompt will open and you will see the installer again, except this one will stay open until you press a button.
                            </p>

                            <br />
                            <Text fontSize="lg" fontWeight="medium">
                                Before Running:
                            </Text>
                            <p>
                                Try running Debug Mode by checking the Debug Mode option and pressing Save Settings.
                            </p>

                            <br />
                            <Text fontSize="lg" fontWeight="medium">
                                Next Steps:
                            </Text>
                            <p>
                                Then, reach out on Discord to see if we can resolve this problem.
                            </p>
                            <br />
                            <Text fontSize="lg" fontWeight="medium">
                                If all else fails:
                            </Text>
                            <p>
                                If all else fails, then proceed. NOTE: You will need a stable internet connection. This process could take a while so hang tight.
                            </p>
                            <br />
                            <Flex alignItems="center">
                                <Text fontSize="lg" fontWeight="medium">Use AMD Installer: </Text>
                                {' '}
                                <Switch size="lg" onChange={() => setUseAMDInstaller(!useAMDInstaller)} isChecked={useAMDInstaller} />
                            </Flex>
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
                                        DebugInstaller();
                                }}>
                                Install Artroom {useAMDInstaller ? 'for AMD' : 'for NVIDIA'}
                            </Button>
                        </AlertDialogFooter>
                    </AlertDialogContent>
                </AlertDialogOverlay>
            </AlertDialog>
        </>
    );
}

export default DebugInstallerModal;
