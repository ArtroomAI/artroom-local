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
    useToast,
    RadioGroup,
    Radio
} from '@chakra-ui/react';
import { useRecoilState } from 'recoil';
import { artroomPathState } from '../../SettingsManager';

function DebugInstallerModal () {
    const { isOpen, onOpen, onClose } = useDisclosure();
    const toast = useToast({});

    const cancelRef = React.useRef();
    const [artroomPath, setArtroomPath] = useRecoilState(artroomPathState);

    const DebugInstaller = () => {
        window.api.pythonInstall(artroomPath);
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
            {/* <Button
                backgroundColor="red.600"
                colorScheme="red"
                onClick={onOpen}>
                Reinstall Artroom Backend
            </Button> */}

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
                                NOTE: To proceed you MUST run Artroom as Admin
                            </Text>
                            <p>
                                If you do not run Artroom as admin, the downloaded backend will not have the permission to unzip and the reinstall will fail. Once you click Install, you will see the progress below. There may be a few seconds of delay before seeing the first tick of progress. 
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
                            </Flex>
                        </AlertDialogBody>

                        <AlertDialogFooter>
                            <Button
                                onClick={onClose}
                                ref={cancelRef}>
                                Cancel
                            </Button>
                        </AlertDialogFooter>
                    </AlertDialogContent>
                </AlertDialogOverlay>
            </AlertDialog>
        </>
    );
}

export default DebugInstallerModal;
