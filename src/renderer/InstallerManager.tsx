import React, { useEffect, useState } from 'react';
import { useRecoilState, useRecoilValue } from 'recoil';
import {
    Modal,
    ModalOverlay,
    ModalContent,
    ModalHeader,
    ModalBody,
    ModalFooter,
    Flex,
    Button,
    Input,
    Radio,
    RadioGroup,
    Text,
    Spacer,
    useToast,
    Checkbox,
    HStack,
} from '@chakra-ui/react';
import { artroomPathState, debugModeState, modelsDirState } from './SettingsManager';
import path from 'path';

export const InstallerManager = () => {
    const toast = useToast({});

    const debugMode = useRecoilValue(debugModeState);

    const [showArtroomInstaller, setShowArtroomInstaller] = useState(false);
    const [artroomPath, setArtroomPath] = useRecoilState(artroomPathState);
    const [modelsDir, setModelsDir] = useRecoilState(modelsDirState)
    const [sameModelDirAndArtroomPath, setSameModelDirAndArtroomPath] = useState(false);
    const [gpuType, setGpuType] = useState('NVIDIA');
    const [downloadMessage, setDownloadMessage] = useState('');
    const [downloading, setDownloading] = useState(false);

    useEffect(() => {
        window.api.runPyTests(artroomPath).then((result) => {
            if (result === 'success\r\n') {
                console.log(result);
                window.api.startArtroom(artroomPath, debugMode);
                setShowArtroomInstaller(false);
                toast({
                    title: 'All Artroom paths & dependencies successfully found!',
                    status: 'success',
                    position: 'top',
                    duration: 2000,
                    isClosable: true
                });
            } else if (result.length > 0) {
                setShowArtroomInstaller(true)
            }
        });
    }, [artroomPath]);

    useEffect(() => {
        window.api.fixButtonProgress((_, str) => {
            setDownloadMessage(str);
            if (str.includes("Finished")) {
                setDownloading(false);
                setShowArtroomInstaller(false);
                window.api.startArtroom(artroomPath, debugMode);
            }
        });
    }, []);

    const handleRunClick = () => {
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
        setDownloading(true);
        try {
            if (sameModelDirAndArtroomPath) {
                setModelsDir(path.join(artroomPath, 'artroom', 'model_weights'))
            }
            window.api.pythonInstall(artroomPath, gpuType);
        } catch (error) {
            console.error(error);
            setDownloading(false);
        }
    }

    function handleSelectArtroomClick() {
        window.api.chooseUploadPath().then(setArtroomPath)
    }
    function handleSelectModelClick() {
        window.api.chooseUploadPath().then(setModelsDir)
    }

    return (
        <Modal 
            size='4xl'
            isOpen={showArtroomInstaller} 
            onClose={() => {}} // Disable closing the modal
            scrollBehavior='outside'
            isCentered
            >
            <ModalOverlay bg='blackAlpha.900'/>
            <ModalContent>
                <ModalHeader>Artroom Installer</ModalHeader>
                <ModalBody>
                    <Text>Enter the path where you want to install Artroom</Text>
                    <Text mb='4'>(Note: This is ~11GB of data. Please make sure your drive has enough storage space)</Text>
                    <Flex flexDirection='row' justifyItems='center' alignItems='center' mb='4'>
                        <Input width="80%" placeholder='Artroom will be saved in YourPath/artroom' value={artroomPath} onChange={(event) => {setArtroomPath(event.target.value)}} mr='4' />
                        <Button onClick={handleSelectArtroomClick}>Select</Button>
                    </Flex>
                    <HStack>
                        <Text mb='1'>Where do you want to keep your models?</Text>
                        <Checkbox isChecked={sameModelDirAndArtroomPath} onChange={() => { setSameModelDirAndArtroomPath(!sameModelDirAndArtroomPath) }}>Use Artroom Path</Checkbox>   
                    </HStack>
                    <Flex flexDirection='row' alignItems='center' mb='4'>
                        <Input 
                            width="80%" 
                            placeholder='Model will be saved in YourPath/artroom/model_weights' 
                            value={modelsDir} 
                            onChange={(event) => {setModelsDir(event.target.value)}} 
                            isDisabled={sameModelDirAndArtroomPath} 
                            mr='4' />
                        <Button onClick={handleSelectModelClick}>Select</Button>
                    </Flex>
                    <RadioGroup value={gpuType} onChange={(event)=>{setGpuType(event)}} mb='4'>
                    <Text mb='4'>Do you have an NVIDIA or AMD GPU?:</Text>
                    <Flex flexDirection='row' alignItems='center'>
                        <Radio value='NVIDIA' mr='2' />
                        NVIDIA
                    </Flex>
                    <Flex flexDirection='row' alignItems='center'>
                        <Radio value='AMD' mr='2' />
                        AMD
                    </Flex>
                    </RadioGroup>
                    {
                    downloadMessage && (
                    <Flex width="100%">
                        <Flex width="100%">Installation progress</Flex>
                        <Spacer/>
                        <Flex width="100%">{downloadMessage}</Flex>
                    </Flex>)
                    }
                </ModalBody>

                <ModalFooter justifyContent='center'>
                    <Button isLoading={downloading} isDisabled={downloading} onClick={handleRunClick}>Install Artroom</Button>
                </ModalFooter>
            </ModalContent>
        </Modal>
    )
}
