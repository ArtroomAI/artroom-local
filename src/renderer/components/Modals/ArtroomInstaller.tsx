import React, { useEffect, useState } from 'react';
import { useRecoilState } from 'recoil';
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
} from '@chakra-ui/react';
import { artroomPathState } from '../../SettingsManager';

function ArtroomInstaller ({showArtroomInstaller, setShowArtroomInstaller}) {

    const [artroomPath, setArtroomPath] = useRecoilState(artroomPathState);
    const [gpuType, setGpuType] = useState('NVIDIA');
    const [downloadMessage, setDownloadMessage] = useState('');
    const [downloading, setDownloading] = useState(false);

    const toast = useToast({});

    useEffect(() => {
        window.api.fixButtonProgress((_, str) => {
            setDownloadMessage(str);
            console.log(str);
        });
    }, []);
    
    useEffect(()=>{
        //Let users change their path if they just move the file
        window.api.runPyTests(artroomPath).then((result) => {
            console.log(result)
            if (result === 'success\r\n') {
                setShowArtroomInstaller(false)
            }
        });
    },[artroomPath])

    function handleRunClick() {
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
        window.api.pythonInstall(artroomPath, gpuType).then(()=>{
            // setDownloading(false)
            // setShowArtroomInstaller(false)
        });

    }

    function handleSelectPathClick() {
        window.api.chooseUploadPath().then(setArtroomPath)
    }

    return (
        <Modal 
            size='6xl'
            isOpen={showArtroomInstaller} 
            onClose={() => {}} // Disable closing the modal
            scrollBehavior='outside'
            isCentered
            >
            <ModalOverlay bg='blackAlpha.900'/>
            <ModalContent>
                <ModalHeader>Artroom Installer</ModalHeader>
                <ModalBody>
                    <Text mb='4'>Enter the path where you want to install Artroom (Note: This is ~11GB of data. Please make sure your drive has enough storage space)</Text>
                    <Flex flexDirection='row' alignItems='center' mb='4'>
                        <Input placeholder='Artroom Path' value={artroomPath} onChange={(event) => {setArtroomPath(event.target.value)}} mr='4' />
                        <Button onClick={handleSelectPathClick}>Select</Button>
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

export default ArtroomInstaller;