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
    useToast,
    Checkbox,
    Link,
    VStack
} from '@chakra-ui/react';
import { artroomPathState, debugModeState, modelsDirState, cloudOnlyState } from './SettingsManager';
import path from 'path';
import { spawn } from 'child_process';

export const InstallerManager = () => {
    const toast = useToast({});

    const debugMode = useRecoilValue(debugModeState);

    const [cloudOnly, setCloudOnly] = useRecoilState(cloudOnlyState);

    const [showArtroomInstaller, setShowArtroomInstaller] = useState(false);
    const [artroomPath, setArtroomPath] = useRecoilState(artroomPathState);
    const [modelsDir, setModelsDir] = useRecoilState(modelsDirState || path.join(artroomPath, 'artroom', 'model_weights'))
    const [customPath, setCustomPath] = useState(false);
    const [downloadMessage, setDownloadMessage] = useState('');
    const [downloading, setDownloading] = useState(false);

    const [realisticStarter, setRealisticStarter] = useState(false);
    const [animeStarter, setAnimeStarter] = useState(false);
    const [landscapesStarter, setLandscapesStarter] = useState(true);
    
    const [stillHavingTrouble, setStillHavingTrouble] = useState(false);
    useEffect(() => {
        if(cloudOnly) return;

        window.api.runPyTests(artroomPath).then((result) => {
            if (result === 'success\r\n' || process.platform == "linux") {
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
        const handlerDiscard = window.api.fixButtonProgress((_: any, str: string) => {
            setDownloadMessage(str);
        });

        return () => {
            handlerDiscard();
        }
    }, []);

    const handleRunClick = async () => {
        toast({
            title: 'Installing Artroom Backend',
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
            let dir = modelsDir;
            if (!customPath) {
                dir = path.join(artroomPath, 'artroom', 'model_weights')
                setModelsDir(dir)
            }
            await window.api.pythonInstall(artroomPath);
            await window.api.downloadStarterModels(dir, realisticStarter, animeStarter, landscapesStarter);
            setDownloading(false);
            setShowArtroomInstaller(false);
            window.api.startArtroom(artroomPath, debugMode);
        } catch (error) {
            console.error(error);
            setDownloading(false);
        }
    }

    const handleBackupDownloader = () => {
        window.api.backupDownload(artroomPath)
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
                <ModalHeader>{`Artroom Installer (~6GB)`}</ModalHeader>
                <ModalBody>
                    <Text>Artroom Engine Backend Install Location</Text>
                    {customPath && <Text mb='4'>{`(Note: Do NOT select the Artroom folder that was installed on startup)`}</Text>
                    }
                    <Flex flexDirection='row' justifyItems='center' alignItems='center' mb='4'>
                        <Input 
                            width="80%" 
                            placeholder='Artroom will be saved in YourPath/artroom' 
                            value={artroomPath} 
                            onChange={(event) => {setArtroomPath(event.target.value)}} 
                            isDisabled={!customPath} 
                            mr='4' />
                        {customPath && <Button onClick={handleSelectArtroomClick}>Select</Button>}
                    </Flex>
                    <Text mb='1'>{`Model Path (This can be changed later in Settings)`}</Text>
                    <Flex flexDirection='row' alignItems='center' mb='4'>
                        <Input 
                            width="80%" 
                            placeholder='Model will be saved in YourPath/artroom/model_weights' 
                            value={modelsDir} 
                            onChange={(event) => {setModelsDir(event.target.value)}} 
                            isDisabled={!customPath} 
                            mr='4' />
                        {customPath && <Button onClick={handleSelectModelClick}>Select</Button>}
                    </Flex>
                    <Checkbox isChecked={customPath} onChange={() => { setCustomPath(!customPath) }}>Use Custom Path</Checkbox>  
                    <Text>
                        {`Do you want a starter model (optional)?`}
                    </Text>
                    <Text mb='4'>
                        {`Want more? Download them to your Artroom folder from `}
                        <Link color='blue.500' href='https://civitai.com/' isExternal>
                            civitai.com
                        </Link>
                    </Text>
                    <VStack alignItems='flex-start'>
                        <Checkbox isChecked={landscapesStarter} onChange={() => { setLandscapesStarter(!landscapesStarter) }}>{`(Popular) DreamShaper `}</Checkbox>  
                        <Checkbox isChecked={realisticStarter} onChange={() => { setRealisticStarter(!realisticStarter) }}>{`(Realistic) ChilloutMix `}</Checkbox>   
                        <Checkbox isChecked={animeStarter} onChange={() => { setAnimeStarter(!animeStarter) }}>{`(Anime) Counterfeit `}</Checkbox>                      
                        <Text 
                            _hover={{
                                cursor: 'pointer',
                                textDecoration: 'underline',
                            }}
                            onClick={() => {
                                handleBackupDownloader()
                                setStillHavingTrouble(true);
                            }}
                            >
                            {`Trouble Downloading? Click Here to launch the backup installer`}
                            </Text>
                            {stillHavingTrouble && 
                                <Text 
                                _hover={{
                                    cursor: 'pointer',
                                    textDecoration: 'underline',
                                }}
                                onClick={() => {
                                    window.api.openInstallTutorial()
                                }}
                                >
                                {`Still Having Trouble Downloading? Click Here to get manual install instructions (Don't worry, it's easy)`}
                                </Text>
                            }

                    </VStack>
                    {
                    downloadMessage && (
                        <Flex width="100%" justifyContent="space-between">
                            <Text>Installation progress</Text>
                            <Text pr='80px'>{downloadMessage}</Text>
                        </Flex>
                    )
                    }
                </ModalBody>

                <ModalFooter justifyContent='center'>
                    <Button isLoading={downloading} isDisabled={downloading} onClick={handleRunClick}>Install Artroom</Button>
                    {/* <Button onClick={() => setCloudOnly(true)}>I want cloud only</Button> */}
                </ModalFooter>
            </ModalContent>
        </Modal>
    )
}
