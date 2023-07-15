import React, { useCallback, useEffect } from 'react';
import { useRecoilState, useSetRecoilState } from 'recoil';
import { initImagePathState, aspectRatioSelectionState } from '../../atoms/atoms';
import {
    Box,
    Image as ChakraImage,
    IconButton,
    ButtonGroup,
    Tooltip,
    VStack,
    FormControl,
    HStack,
    FormLabel,
    Spacer,
    Slider,
    SliderTrack,
    SliderFilledTrack,
    SliderThumb
} from '@chakra-ui/react';
import {
    FiUpload
} from 'react-icons/fi';
import {
    FaTrashAlt, FaClipboardList, FaQuestionCircle
} from 'react-icons/fa';
import { aspectRatioState, initImageState, strengthState } from '../../SettingsManager';
import { getImageDimensions } from '../Utils/image';
import { useDropzone } from 'react-dropzone';

const DragDropFile = () => {
    const [initImagePath, setInitImagePath] = useRecoilState(initImagePathState);
    const [aspectRatioSelection, setAspectRatioSelection] = useRecoilState(aspectRatioSelectionState);
    const [initImage, setInitImage] = useRecoilState(initImageState);
    const setAspectRatio = useSetRecoilState(aspectRatioState);
    const [strength, setStrength] = useRecoilState(strengthState);

    const onDrop = useCallback((acceptedFiles: File[]) => {
        handleFile(acceptedFiles[0]);
    }, []);

    const { getRootProps, getInputProps, open } = useDropzone({ onDrop, useFsAccessApi: false });

    useEffect(() => {
        if (initImagePath) {
            console.log(initImagePath);
            window.api.getImageFromPath(initImagePath).then(({ b64 }) => {
                getImageDimensions(b64).then(({ width, height }) => {
                    setInitImage(b64);
                    if (aspectRatioSelection === "Init Image") {
                        setAspectRatio(`${width}:${height}`);
                    }
                });
            });
        }
        // update only when ImagePath is changed - prevents changing settings infinitely
        // eslint-disable-next-line react-hooks/exhaustive-deps
      }, [initImagePath]);

    const handleFile = useCallback((file: File) => {
        if (file.type === "image/jpeg" || file.type === "image/png" || file.type === "image/heic") {
            console.log(file.path);
            setInitImagePath(file.path);
        } else {
            console.log("Invalid file type. Please select an image file (jpg, png or heic).");
        }
    }, []);

    return (
        <VStack>
            { initImage.length > 0 &&
                <FormControl className="strength-input">
                    <HStack align='center'>
                        <FormLabel htmlFor="Strength">
                            Strength:
                        </FormLabel>
                        <Spacer />
                        <Tooltip
                            fontSize="md"
                            label="Strength determines how much your output will resemble your input image. Closer to 0 means it will look more like the original and closer to 1 means use more noise and make it look less like the input"
                            placement="left"
                            shouldWrapChildren
                        >
                            <FaQuestionCircle color="#777" />
                        </Tooltip>
                    </HStack>
                    <Slider
                        defaultValue={0.75}
                        id="strength"
                        max={1}
                        min={0.01}
                        name="strength"
                        onChange={setStrength}        
                        step={0.01}
                        value={strength}
                        variant="outline"
                    >
                        <SliderTrack bg="#EEEEEE">
                            <SliderFilledTrack bg="#4f8ff8" />
                        </SliderTrack>
    
                        <Tooltip
                            bg="#4f8ff8"
                            color="white"
                            isOpen={true}
                            label={`${strength}`}
                            placement="right"
                        >
                            <SliderThumb />
                        </Tooltip>
                    </Slider>
                </FormControl> 
            }

            <Box
                bg="#080B16"
                height="180px"
                width="180px"
                {...getRootProps()}
            >
                <Box
                    border="1px"
                    borderStyle="ridge"
                    height="180px"
                    rounded="md"
                    style={{ display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        textAlign: 'center',
                        borderColor: '#FFFFFF20' }}
                    width="180px"
                >
                    {initImage.length > 0 ?
                        <ChakraImage
                            boxSize="178px"
                            fit="contain"
                            rounded="md"
                            src={initImage}
                        />                     
                        : <>
                            Click or Drag A Starting Image Here
                            <Tooltip
                                fontSize="md"
                                label="Upload an image to use as the starting point for your generation instead of just random noise"
                                placement="top"
                                shouldWrapChildren>
                                <FaQuestionCircle color="#777" />
                            </Tooltip>
                        </>
                    }
                </Box>
                <input
                    {...getInputProps({
                        id: "input-file-upload",
                        accept: "image/png, image/jpeg",
                        "multiple": false })}
                    />
            </Box>
            <ButtonGroup width="100%" isAttached variant="outline">
                <Tooltip label="Paste from Clipboard">
                    <IconButton
                        width="45px"
                        aria-label="Paste from Clipboard"
                        border="2px"
                        icon={<FaClipboardList />}
                        onClick={() => {
                            navigator.clipboard.read().then((data) => {
                                const clip = data[0];
                                if(clip && (clip.types.includes('image/png') || clip.types.includes('image/jpeg'))) {
                                    clip.getType('image/png').then((blob) => {
                                        const reader = new FileReader();
                                        reader.readAsDataURL(blob);
                                        reader.onloadend = () => {
                                            const base64data = reader.result;
                                            if(base64data === null) {
                                                return;
                                            }
                                            if(typeof base64data === 'string') {
                                                setInitImage(base64data);
                                            } else {
                                                const enc = new TextDecoder("utf-8");
                                                setInitImage(enc.decode(new Uint8Array(base64data)));
                                            }
                                        };
                                    });
                                }
                            });
                        }}
                    />
                </Tooltip>
                <Tooltip label="Upload">
                    <IconButton
                        border="2px"
                        icon={<FiUpload />}
                        onClick={open}
                        width="90px"
                        aria-label="upload"
                    />
                </Tooltip>
                <Tooltip label="Clear Starting Image">
                    <IconButton
                        aria-label="Clear Starting Image"
                        border="2px"
                        icon={<FaTrashAlt />}
                        width="45px"
                        onClick={() => {
                            setInitImagePath("");
                            setInitImage("");
                            if (aspectRatioSelection === "Init Image") {
                                setAspectRatio("None");
                                setAspectRatioSelection("None");
                            }
                        }}
                    />
                </Tooltip>
            </ButtonGroup>
        </VStack>
    );
};

export default DragDropFile;
