import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useRecoilState, useSetRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import {
    Box,
    Image as ChakraImage,
    IconButton,
    ButtonGroup,
    Tooltip
} from '@chakra-ui/react';
import {
    FiUpload
} from 'react-icons/fi';
import {
    FaTrashAlt, FaClipboardList
} from 'react-icons/fa';
import { aspectRatioState, heightState, initImageState, widthState } from '../../SettingsManager';

const getImageDimensions = (base64: string) => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
        img.onerror = reject;
        img.src = base64;
    });
};

const DragDropFile = () => {
    const [dragActive, setDragActive] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);
    const [initImagePath, setInitImagePath] = useRecoilState(atom.initImagePathState);
    const [aspectRatioSelection, setAspectRatioSelection] = useRecoilState(atom.aspectRatioSelectionState);
    const [initImage, setInitImage] = useRecoilState(initImageState);
    const setWidth = useSetRecoilState(widthState);
    const setHeight = useSetRecoilState(heightState);
    const setAspectRatio = useSetRecoilState(aspectRatioState);

    useEffect(() => {
        if (initImagePath) {
            console.log(initImagePath);
            window.api.getImageFromPath(initImagePath).then((result) => {
                getImageDimensions(result.b64).then((dimensions) => {
                    setInitImage(result.b64);
                    if (aspectRatioSelection === "Init Image") {
                        setWidth(dimensions.width);
                        setHeight(dimensions.height);
                    }
                });
            });
        }
        // update only when ImagePath is changed - prevents changing settings infinitely
        // eslint-disable-next-line react-hooks/exhaustive-deps
      }, [initImagePath]);

    // Handle drag events
    const handleDrag: React.DragEventHandler<HTMLElement> = function (e) {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    // Triggers when file is dropped
    const handleDrop: React.DragEventHandler<HTMLDivElement> = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    }, []);

    const handleFile = useCallback((file: File) => {
        if (file.type === "image/jpeg" || file.type === "image/png" || file.type === "image/heic") {
            console.log(file.path);
            setInitImagePath(file.path);
        } else {
            console.log("Invalid file type. Please select an image file (jpg, png or heic).");
        }
    }, []);

    // Triggers when file is selected with click
    const handleChange: React.ChangeEventHandler<HTMLInputElement> = function (e) {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    };

    // Triggers the input when the button is clicked
    const onButtonClick = () => {
        inputRef.current.click();
    };

    return (
        <Box
            bg="#080B16"
            height="140px"
            width="140px"
        >
            {initImage.length > 0
                ? <Box
                    border="1px"
                    borderStyle="ridge"
                    height="140px"
                    onClick={onButtonClick}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    rounded="md"
                    style={{ display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        textAlign: 'center',
                        borderColor: '#FFFFFF20' }}
                    width="140px"
                >
                    <ChakraImage
                        boxSize="140px"
                        fit="contain"
                        rounded="md"
                        src={initImage}
                    />
                </Box>
                : <Box
                    border="1px"
                    borderStyle="ridge"
                    height="140px"
                    onClick={onButtonClick}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    px={4}
                    rounded="md"
                    style={{ display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        textAlign: 'center',
                        borderColor: '#FFFFFF20' }}
                    width="140px"
                >
                    Click or Drag Image Here
                </Box> }

            <form
                id="form-file-upload"
                onDragEnter={handleDrag}
                onSubmit={(e) => e.preventDefault()}
            >
                <input
                    accept="image/png, image/jpeg"
                    id="input-file-upload"
                    multiple={false}
                    onChange={handleChange}
                    ref={inputRef}
                    type="file"
                />

                <label
                    htmlFor="input-file-upload"
                    id="label-file-upload"
                >

                <ButtonGroup pt={2} isAttached variant="outline">
                <Tooltip label="Clear Init Image">
                    <IconButton
                    aria-label="Clear Init Image"
                    border="2px"
                    icon={<FaClipboardList />}
                    onClick={() => {
                        navigator.clipboard.read().then((data) => {
                          for (let i = 0; i < data.length; i++) {
                            if (data[i].types.includes('image/png') || data[i].types.includes('image/jpeg')) {
                              data[i].getType('image/png').then((blob) => {
                                const reader = new FileReader();
                                reader.readAsDataURL(blob);
                                reader.onloadend = () => {
                                  const base64data = reader.result;
                                 setInitImage(base64data);
                                };
                              });
                              break;
                            }
                          }
                        });
                      }}
                    />
                </Tooltip>
                <Tooltip label="Upload">
                    <IconButton
                    border="2px"
                    icon={<FiUpload />}
                    onClick={onButtonClick}
                    width="65px"
                    aria-label="upload"
                    />
                </Tooltip>
                <Tooltip label="Clear Init Image">
                    <IconButton
                    aria-label="Clear Init Image"
                    border="2px"
                    icon={<FaTrashAlt />}
                    onClick={() => {
                        inputRef.current.value = null;
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
                </label>
            </form>
        </Box>
    );
};

export default DragDropFile;
