import React, { useState, useEffect, useCallback } from 'react';
import { useRecoilState } from 'recoil';
import {
    Box,
    Button,
    Checkbox,
    Flex,
    FormControl,
    FormLabel,
    Input,
    VStack,
    HStack,
    Tooltip,
    Radio,
    RadioGroup,
    Stack,
    Spacer,
    useToast
} from '@chakra-ui/react';
import {
    FaQuestionCircle
} from 'react-icons/fa';
import DebugInstallerModal from './Modals/DebugInstallerModal';
import { highresFixState, imageSavePathState, longSavePathState, modelsDirState, saveGridState, speedState, showIntermediatesState, artroomPathState, debugModeState } from '../SettingsManager';

function Settings () {
    const toast = useToast({});

    const [debugMode, setDebugMode] = useRecoilState(debugModeState);
    const [longSavePath, setLongSavePath] = useRecoilState(longSavePathState);
    const [highresFix, setHighresFix] = useRecoilState(highresFixState);
    const [showIntermediates, setShowIntermediates] = useRecoilState(showIntermediatesState);
    const [speed, setSpeed] = useRecoilState(speedState);
    const [imageSavePath, setImageSavePath] = useRecoilState(imageSavePathState);
    const [saveGrid, setSaveGrid] = useRecoilState(saveGridState);
    const [modelsDir, setModelsDir] = useRecoilState(modelsDirState);

    const [debugModeTemp, setDebugModeTemp] = useState(debugMode);
    const [longSavePathTemp, setLongSavePathTemp] = useState(longSavePath);
    const [highresFixTemp, setHighresFixTemp] = useState(highresFix);
    const [showIntermediatesTemp, setShowIntermediatesTemp] = useState(showIntermediates);
    const [speedTemp, setSpeedTemp] = useState(speed);
    const [imageSavePathTemp, setImageSavePathTemp] = useState(imageSavePath);
    const [saveGridTemp, setSaveGridTemp] = useState(saveGrid);
    const [modelsDirTemp, setModelsDirTemp] = useState(modelsDir);
    
    const [artroomPath, setArtroomPath] = useRecoilState(artroomPathState);
    const [downloadMessage, setDownloadMessage] = useState('');

    // load defaults
    useEffect(() => {
        console.log(modelsDir);
        setDebugModeTemp(debugMode);
        setLongSavePathTemp(longSavePath);
        setHighresFixTemp(highresFix);
        setShowIntermediatesTemp(showIntermediates);
        setSpeedTemp(speed);
        setImageSavePathTemp(imageSavePath);
        setSaveGridTemp(saveGrid);
        setModelsDirTemp(modelsDir);
    }, []);

    const save = useCallback(() => {
        if(debugModeTemp !== debugMode) {  
            if(debugModeTemp) {
                toast({
                    id: 'restart-server-debug-on',
                    title: 'Resetting Artroom with Debug Mode on',
                    status: 'info',
                    position: 'top',
                    duration: 7000,
                    isClosable: true
                });
            } else {
                toast({
                    id: 'restart-server-debug-off',
                    title: 'Resetting Artroom with Debug Mode off',
                    status: 'info',
                    position: 'top',
                    duration: 7000,
                    isClosable: true
                });
            }
            window.api.restartServer(artroomPath, debugModeTemp);
        }
        setDebugMode(debugModeTemp);
        setLongSavePath(longSavePathTemp);
        setHighresFix(highresFixTemp);
        setShowIntermediates(showIntermediatesTemp);
        setSpeed(speedTemp);
        setImageSavePath(imageSavePathTemp);
        setSaveGrid(saveGridTemp);
        setModelsDir(modelsDirTemp);
        toast({
            title: 'Settings have been updated',
            status: 'success',
            position: 'top',
            duration: 1500,
            isClosable: false,
            containerStyle: {
                pointerEvents: 'none'
            }
        });
    }, [toast, debugModeTemp, highresFixTemp, imageSavePathTemp, longSavePathTemp, modelsDirTemp, saveGridTemp, speedTemp]);

    useEffect(() => {
        const handlerDiscard = window.api.fixButtonProgress((_, str) => {
            setDownloadMessage(str);
            console.log(str);
        });

        return () => {
            handlerDiscard();
        }
    }, []);

    const chooseUploadPath = () => {
        window.api.chooseUploadPath().then(setImageSavePathTemp);
    };

    const chooseCkptDir = () => {
        window.api.chooseUploadPath().then(setModelsDirTemp);
    };

    return (
        <Box
            height="90%"
            ml="30px"
            p={5}
            rounded="md"
            width="75%">
            <VStack
                align="flex-start"
                className="settings"
                spacing={5}>
                <FormControl
                    className="image-save-path-input"
                    width="full">
                    <FormLabel htmlFor="image_save_path">
                        Image Output Folder
                    </FormLabel>

                    <HStack>
                        <Input
                            id="image_save_path"
                            name="image_save_path"
                            onChange={(event) => setImageSavePathTemp(event.target.value)}
                            type="text"
                            value={imageSavePathTemp}
                            variant="outline"
                        />

                        <Button onClick={chooseUploadPath}>
                            Choose
                        </Button>
                    </HStack>
                </FormControl>

                <FormControl
                    className="model-ckpt-dir-input"
                    width="full">
                    <HStack>
                        <Tooltip
                            fontSize="md"
                            label="When making folders, choose a non-root directory (so do E:/Models instead of E:/)"
                            placement="top"
                            shouldWrapChildren>
                            <FaQuestionCircle color="#777" />
                        </Tooltip>

                        <FormLabel htmlFor="ckpt_dir">
                            Model Weights Folder
                        </FormLabel>
                    </HStack>

                    <HStack>
                        <Input
                            id="ckpt_dir"
                            name="ckpt_dir"
                            onChange={(event) => setModelsDirTemp(event.target.value)}
                            type="text"
                            value={modelsDirTemp}
                            variant="outline"
                        />

                        <Button onClick={chooseCkptDir}>
                            Choose
                        </Button>
                    </HStack>
                </FormControl>

                <FormControl className="speed-input">
                    <HStack>
                        <Tooltip
                            fontSize="md"
                            label="Generate faster but use more GPU memory. Be careful of OOM (Out of Memory) error"
                            mt="3"
                            placement="right"
                            shouldWrapChildren>
                            <FaQuestionCircle color="#777" />
                        </Tooltip>

                        <FormLabel htmlFor="Speed">
                            Choose Generation Speed
                        </FormLabel>
                    </HStack>

                    <RadioGroup
                        id="speed"
                        name="speed"
                        onChange={setSpeedTemp}
                        value={speedTemp}>
                        <Stack
                            direction="row"
                            spacing="20">
                            ``
                            <Radio value="Low">
                                Low
                            </Radio>

                            <Radio value="Medium">
                                Medium
                            </Radio>

                            <Radio value="High">
                                High
                            </Radio>
                        </Stack>
                    </RadioGroup>
                </FormControl>

                <HStack className="show-intermediates-input">
                    <Checkbox
                        id="show_intermediates"
                        isChecked={showIntermediatesTemp}
                        name="show_intermediates"
                        onChange={() => {
                            setShowIntermediatesTemp((si) => !si);
                        }}
                    >
                        Show Intermediates
                    </Checkbox>

                    <Tooltip
                        fontSize="md"
                        label="Show intermediate images while generating"
                        mt="3"
                        placement="right"
                        shouldWrapChildren>
                        <FaQuestionCircle color="#777" />
                    </Tooltip>
                </HStack>

                <HStack className="highres-fix-input">
                    <Checkbox
                        id="highres_fix"
                        isChecked={highresFixTemp}
                        name="highres_fix"
                        onChange={() => {
                            setHighresFixTemp((hf) => !hf);
                        }}
                    >
                        Use Highres Fix
                    </Checkbox>

                    <Tooltip
                        fontSize="md"
                        label="Once you get past 1024x1024, will generate a smaller image, upscale, and then img2img to improve quality"
                        mt="3"
                        placement="right"
                        shouldWrapChildren>
                        <FaQuestionCircle color="#777" />
                    </Tooltip>
                </HStack>

                <HStack className="long-save-path-input">
                    <Checkbox
                        id="long_save_path"
                        isChecked={longSavePathTemp}
                        name="long_save_path"
                        onChange={() => {
                            setLongSavePathTemp((lsp) => !lsp);
                        }}
                    >
                        Use Long Save Path
                    </Checkbox>

                    <Tooltip
                        fontSize="md"
                        label="Put the images in a folder named after the prompt (matches legacy Artroom naming conventions)"
                        mt="3"
                        placement="right"
                        shouldWrapChildren>
                        <FaQuestionCircle color="#777" />
                    </Tooltip>
                </HStack>

                <HStack className="save-grid-input">
                    <Checkbox
                        id="save_grid"
                        isChecked={saveGridTemp}
                        name="save_grid"
                        onChange={() => {
                            setSaveGridTemp((sg) => !sg);
                        }}
                    >
                        Save Grid
                    </Checkbox>

                    <Tooltip
                        fontSize="md"
                        label="Save batch in one big grid image"
                        mt="3"
                        placement="right"
                        shouldWrapChildren>
                        <FaQuestionCircle color="#777" />
                    </Tooltip>
                </HStack>

                <HStack className="debug-mode-input">
                    <Checkbox
                        id="debug_mode"
                        isChecked={debugModeTemp}
                        name="debug_mode"
                        onChange={() => {
                            setDebugModeTemp((dm) => !dm);
                        }}
                    >
                        Debug Mode
                    </Checkbox>

                    <Tooltip
                        fontSize="md"
                        label="Opens cmd console of detailed outputs during image generation"
                        mt="3"
                        placement="right"
                        shouldWrapChildren>
                        <FaQuestionCircle color="#777" />
                    </Tooltip>
                </HStack>

                <Flex width="100%">
                    <Button
                        alignContent="center"
                        className="save-settings-button"
                        onClick={save}>
                        Save Settings
                    </Button>
                    <Spacer/>
                    {/* <DebugInstallerModal/> */}
                    <Button
                        backgroundColor="red.600"
                        colorScheme="red"
                        alignContent="center"
                        className="reinstall-python-dependencies"
                        onClick={()=>{window.api.pythonInstallDependencies(artroomPath)}}>
                        Update Packages
                    </Button>
                </Flex>
                
                {
                    downloadMessage && (
                        <Flex width="100%">
                            <Flex width="100%">Installation progress</Flex>
                            <Spacer/>
                            <Flex width="100%">{downloadMessage}</Flex>
                        </Flex>)
                }
            </VStack>
        </Box>
    );
}

export default Settings;
