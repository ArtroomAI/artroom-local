import React, { useState, useEffect } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import axios from 'axios';
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
    Select,
    NumberInput,
    NumberInputField,
    createStandaloneToast,
    Spacer
} from '@chakra-ui/react';
import {
    FaQuestionCircle
} from 'react-icons/fa';
import DebugInstallerModal from './Modals/DebugInstallerModal';

function Settings () {
    const { ToastContainer, toast } = createStandaloneToast();

    const [image_save_path, setImageSavePath] = useRecoilState(atom.imageSavePathState);
    const [long_save_path, setLongSavePath] = useRecoilState(atom.longSavePathState);
    const [highres_fix, setHighresFix] = useRecoilState(atom.highresFixState);
    const [speed, setSpeed] = useRecoilState(atom.speedState);
    const [save_grid, setSaveGrid] = useRecoilState(atom.saveGridState);
    const [debug_mode, setDebugMode] = useRecoilState(atom.debugMode);
    const [ckpt_dir, setCkptDir] = useRecoilState(atom.ckptDirState);
    const [delay, setDelay] = useRecoilState(atom.delayState);
    const [vae, setVae] = useRecoilState(atom.vaeState);

    const [debug_mode_orig, setDebugModeOrig] = useState(true);

    useEffect(
        () => {
            window.api.getSettings().then((result) => {
                const settings = JSON.parse(result);
                setImageSavePath(settings.image_save_path);
                setLongSavePath(settings.long_save_path);
                setHighresFix(settings.highres_fix);
                setCkptDir(settings.ckpt_dir);
                setSpeed(settings.speed);
                setDelay(settings.delay);
                setSaveGrid(settings.save_grid);
                setDebugMode(settings.debug_mode);
                setDebugModeOrig(settings.debug_mode);
            });
        },
        []
    );


    const saveSettings = () => {
        setDebugModeOrig(debug_mode);
        const output = {
            speed,
            image_save_path,
            long_save_path,
            highres_fix,
            ckpt_dir,
            save_grid,
            debug_mode,
            delay
        };
        axios.post(
            'http://127.0.0.1:5300/update_settings',
            output,
            {
                headers: { 'Content-Type': 'application/json' }
            }
        ).then(() => {
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
        });
    };

    const failedFlaskRestartMessage = () => {
        toast({
            id: 'restart-server-failed',
            title: 'Flask server did not restart properly please toggle Debug Mode setting and retry',
            status: 'error',
            position: 'top',
            duration: 7000,
            isClosable: false
        });
    };

    const restartFlask = () => {
        window.api.restartServer(debug_mode).then((result) => {
            if (result === 200) {
                saveSettings();
                console.log(`Success, Debug Mode now: ${debug_mode}`);
            } else {
                failedFlaskRestartMessage();
            }
        });
    };

    const submitEvent = () => {
        if (debug_mode === true && debug_mode_orig === false) {
            // Restart server and turn debug mode on
            toast({
                id: 'restart-server-debug-on',
                title: 'Resetting Artroom with Debug Mode on',
                status: 'info',
                position: 'top',
                duration: 7000,
                isClosable: true
            });
            restartFlask(); // Restarts flask server first, then save settings
        } else if (debug_mode === false && debug_mode_orig === true) {
            // Restart server and turn debug mode off
            toast({
                id: 'restart-server-debug-off',
                title: 'Resetting Artroom with Debug Mode off',
                status: 'info',
                position: 'top',
                duration: 7000,
                isClosable: true
            });
            restartFlask(); // Restarts flask server first, then save settings
        } else {
            // Just save settings as normal
            saveSettings();
        }
    };

    const chooseUploadPath = () => {
        window.api.chooseUploadPath().then(setImageSavePath);
    };

    const chooseCkptDir = () => {
        window.api.chooseUploadPath().then(setCkptDir);
    };
    
    const chooseVae = () => {
        window.api.chooseVae().then(setVae);
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
                            onChange={(event) => setImageSavePath(event.target.value)}
                            type="text"
                            value={image_save_path}
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
                            onChange={(event) => setCkptDir(event.target.value)}
                            type="text"
                            value={ckpt_dir}
                            variant="outline"
                        />

                        <Button onClick={chooseCkptDir}>
                            Choose
                        </Button>
                    </HStack>
                </FormControl>

                <FormControl
                    className="model-vae-input"
                    width="full">
                    <HStack>
                        <Tooltip
                            fontSize="md"
                            label="Choose vae file"
                            placement="top"
                            shouldWrapChildren>
                            <FaQuestionCircle color="#777" />
                        </Tooltip>

                        <FormLabel htmlFor="vae">
                            Vae file
                        </FormLabel>
                    </HStack>

                    <HStack>
                        <Input
                            id="vae"
                            name="vae"
                            onChange={(event) => setVae(event.target.value)}
                            type="text"
                            value={vae}
                            variant="outline"
                        />

                        <Button onClick={chooseVae}>
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
                        onChange={setSpeed}
                        value={speed}>
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

                            <Radio value="Max">
                                Max (experimental)
                            </Radio>
                        </Stack>
                    </RadioGroup>
                </FormControl>

                <FormControl className="queue-delay-input">
                    <HStack>
                        <Tooltip
                            fontSize="md"
                            label="Add a delay between runs. Mostly to prevent GPU from getting too hot if it's nonstop. Runs on next Queue start (need to Stop and Start queue again)."
                            placement="top"
                            shouldWrapChildren>
                            <FaQuestionCircle color="#777" />
                        </Tooltip>

                        <FormLabel htmlFor="delay">
                            Queue Delay
                        </FormLabel>
                    </HStack>

                    <NumberInput
                        id="delay"
                        min={1}
                        name="delay"
                        onChange={(v, n) => {
                            setDelay(n);
                        }}
                        step={1}
                        value={delay}
                        variant="outline"
                        w="150px"
                    >
                        <NumberInputField id="delay" />
                    </NumberInput>
                </FormControl>

                <HStack className="highres-fix-input">
                    <Checkbox
                        id="highres_fix"
                        isChecked={highres_fix}
                        name="highres_fix"
                        onChange={() => {
                            setHighresFix(!highres_fix);
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
                        isChecked={long_save_path}
                        name="long_save_path"
                        onChange={() => {
                            setLongSavePath(!long_save_path);
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
                        isChecked={save_grid}
                        name="save_grid"
                        onChange={() => {
                            setSaveGrid(!save_grid);
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
                        isChecked={debug_mode}
                        name="debug_mode"
                        onChange={() => {
                            setDebugMode(!debug_mode);
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
                        onClick={submitEvent}>
                        Save Settings
                    </Button>

                    {/* <Spacer></Spacer>
                <DebugInstallerModal/> */}
                </Flex>
            </VStack>
        </Box>
    );
}

export default Settings;