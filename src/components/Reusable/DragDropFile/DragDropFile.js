import React, { useEffect, useRef, useState } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../../atoms/atoms';
import { 
    Box, 
    Image,
    IconButton,
    ButtonGroup
} from '@chakra-ui/react';
import {
    FiUpload,
} from 'react-icons/fi'
import {
    FaTrashAlt,
  } from 'react-icons/fa'

const DragDropFile = () => {
    const [dragActive, setDragActive] = useState(false);
    const inputRef = useRef(null);
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);
    const [initImagePath, setInitImagePath] = useRecoilState(atom.initImagePathState);

    function getImageFromPath(){
        console.log(initImagePath);
        if (initImagePath.length > 0){
          window['getImageFromPath'](initImagePath).then((result) => {
            setInitImage(result);
        });
        }
    }

    useEffect(()=>{
        getImageFromPath();
    },[initImagePath])

    // handle drag events
    const handleDrag = function(e) {
      e.preventDefault();
      e.stopPropagation();
      if (e.type === "dragenter" || e.type === "dragover") {
        setDragActive(true);
      } else if (e.type === "dragleave") {
        setDragActive(false);
      }
    };
    
    // triggers when file is dropped
    const handleDrop = function(e) {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files);
      }
    };

    const handleFile = function(e) {
        console.log(e[0].path);
        setInitImagePath(e[0].path);
    };
    
    // triggers when file is selected with click
    const handleChange = function(e) {
      e.preventDefault();
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files);
      }
    };
    
  // triggers the input when the button is clicked
    const onButtonClick = () => {
      inputRef.current.click();
    };
    
    return (
        <Box onClick={onButtonClick} width='150px' height='150px' bg='#080B16' onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}>
            {init_image.length > 0 ? 
              <Image boxSize='150px' fit='contain' src={init_image}/>
            : 
              <>
                <Box onClick={onButtonClick} onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop} width='150px' height='150px'/>
              </> 
            }
            
            <form id="form-file-upload" onDragEnter={handleDrag} onSubmit={(e) => e.preventDefault()}>
                <input ref={inputRef} type="file" id="input-file-upload" multiple={false} accept="image/png, image/jpeg" onChange={handleChange}/>
                <label id="label-file-upload" htmlFor="input-file-upload">
                    <ButtonGroup variant='outline' isAttached>
                        <IconButton onClick={onButtonClick} border={'2px'} icon={<FiUpload/>} width="100px"/>
                        <IconButton border={'2px'} onClick = {(event) => 
                          { 
                            setInitImagePath('');
                            setInitImage('');
                          }} 
                          aria-label='Clear Init Image' icon={<FaTrashAlt/>}></IconButton>
                    </ButtonGroup>
                </label>
            </form>
        </Box>
    );
};

export default DragDropFile;