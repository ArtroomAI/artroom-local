import React, { useEffect, useRef, useState } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import {
    Box,
    Image,
    IconButton,
    ButtonGroup
} from '@chakra-ui/react';
import {
    FiUpload
} from 'react-icons/fi';
import {
    FaTrashAlt
} from 'react-icons/fa';

const DragDropFile = () => {
    const [dragActive, setDragActive] = useState(false);
    const inputRef = useRef(null);
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);
    const [initImagePath, setInitImagePath] = useRecoilState(atom.initImagePathState);

    function getImageFromPath () {
        if (initImagePath.length > 0) {
            console.log(initImagePath);
            window.getImageFromPath(initImagePath).then((result) => {
                setInitImage(result);
            });
        }
    }

    useEffect(
        () => {
            getImageFromPath();
        },
        [initImagePath]
    );

    // Handle drag events
    const handleDrag = function (e) {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    // Triggers when file is dropped
    const handleDrop = function (e) {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files);
        }
    };

    const handleFile = function (e) {
        console.log(e[0].path);
        setInitImagePath(e[0].path);
    };

    // Triggers when file is selected with click
    const handleChange = function (e) {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files);
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
            {init_image.length > 0
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
                    <Image
                        boxSize="140px"
                        fit="contain"
                        rounded="md"
                        src={init_image}
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
                    <ButtonGroup
                        isAttached
                        variant="outline"
                    >
                        <IconButton
                            border="2px"
                            icon={<FiUpload />}
                            onClick={onButtonClick}
                            width="100px"
                        />

                        <IconButton
                            aria-label="Clear Init Image"
                            border="2px"
                            icon={<FaTrashAlt />}
                            onClick={(event) => {
                                setInitImagePath('');
                                setInitImage('');
                            }}
                        />
                    </ButtonGroup>
                </label>
            </form>
        </Box>
    );
};

export default DragDropFile;
