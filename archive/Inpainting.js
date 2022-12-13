import {React, useState, useEffect, useRef} from 'react';
import {useRecoilState} from 'recoil';
import * as atom from '../src/atoms/atoms';
// import Logo from "../images/ArtroomLogoTransparent.png";
// import LoadingGif from "../images/loading.gif";
import Prompt from '../src/components/Prompt';
import { ReactSketchCanvas } from 'react-sketch-canvas';
import {
  VStack,
  HStack,
  IconButton,
  NumberInput,
  NumberInputField,
  Box,
  Button,
  Flex,
  Tooltip,
  Checkbox,
  createStandaloneToast
} from "@chakra-ui/react";
import {
  FaPen,
  FaPlus,
  FaMinus,
  FaTrash,
  FaUndoAlt,
  FaRedoAlt,
  FaQuestionCircle
} from 'react-icons/fa'
import {
  BsEraserFill
} from 'react-icons/bs'
import {
  IoColorPalette
} from 'react-icons/io5'
import {
  Colorful
} from '@uiw/react-color';

  function Inpainting() {
    const { ToastContainer, toast } = createStandaloneToast()
    const styles = {
      border: '0.0625rem solid #9c9c9c',
      borderRadius: '0.25rem',
      cursor: 'url(../images/pencil.png)'
    };

    const canvas = useRef(null);

    const [hex, setHex] = useState("#000");
    const [showColorPicker, setShowColorPicker] = useState(false);
    const [eraser, setEraser] = useState(false);  
    const [toolSize, setToolSize] = useState(10);
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);
    const [imageB64, setImageB64] = useRecoilState(atom.imageB64State);

    const [width, setWidth] = useRecoilState(atom.widthState);
    const [height, setHeight] = useRecoilState(atom.heightState);  
    const [text_prompts, setTextPrompts] = useRecoilState(atom.textPromptsState);
    const [negative_prompts, setNegativePrompts] = useRecoilState(atom.negativePromptsState);
    const [batch_name, setBatchName] = useRecoilState(atom.batchNameState);
    const [steps, setSteps]  = useRecoilState(atom.stepsState);
    const [aspect_ratio, setAspectRatio]  = useRecoilState(atom.aspectRatioState);
    const [seed, setSeed] = useRecoilState(atom.seedState);
    const [use_random_seed, setUseRandomSeed] = useRecoilState(atom.useRandomSeedState);
    const [n_samples, setNSamples] = useRecoilState(atom.nSamplesState);
    const [n_iter, setNIter] = useRecoilState(atom.nIterState);
    const [sampler, setSampler] = useRecoilState(atom.samplerState);
    const [cfg_scale, setCFGScale] = useRecoilState(atom.CFGScaleState);
    const [strength, setStrength] = useRecoilState(atom.strengthState);

    const [queueRunning, setRunning] = useRecoilState(atom.runningState);
    const [runningInpainting, setRunningInpainting] = useRecoilState(atom.runningInpaintingState);
    const [reverse_mask, setReverseMask] = useState(false);
    const [open_pictures, setOpenPictures] = useRecoilState(atom.openPicturesState);
    const [keep_warm, setKeepWarm] = useRecoilState(atom.keepWarmState);

    useEffect(() => {
      if (init_image.length > 0){
        window['getImageB64'](init_image).then((result) => {
          setImageB64(result);
        });
      }
      else{
          setImageB64("");
      }
    },[init_image]);

    // Mask Runner
    useEffect(() => {
      if (!queueRunning || keep_warm){
        if (runningInpainting && imageB64.length === 0){
          setQueueRunning(true);
          canvas.current.exportImage("png").then(image => {
            window['getImageB64'](init_image).then((result) => {
              setImageB64(result);
            })
            if (keep_warm){
              toast({
                title: 'Added to queue',
                status: 'success',
                position: 'top',
                duration: 1500,
                isClosable: false,
                containerStyle: {
                  pointerEvents: 'none'
                },
              })   
            }
            window['paintSD'](image).then((result) => {
              let error_code = result["error_code"]
              if (error_code === "CUDA"){
                toast({
                  title: 'CUDA OOM',
                  description: "Ran out of VRAM loading/running model, try decreasing size of image or lowering generation speed", 
                  status: 'error',
                  position: 'top',
                  duration: null,
                  isClosable: true,
                })   
              }
              else if (error_code === "UNKNOWN"){
                toast({
                  title: 'CUDA OOM',
                  description: "Ran out of VRAM loading/running model, try decreasing size of image or lowering generation speed", 
                  status: 'error',
                  position: 'top',
                  duration: null,
                  isClosable: true,
                })   
              }
              else{
                if (open_pictures){
                  window['getImageDir']();
                }              
              }
              setRunningInpainting(false);
              setQueueRunning(false);
              });              
            }
          )};
        }
        }, [imageB64, runningInpainting]);
      

    const submitInpainting = event => {
        if (!queueRunning && !runningInpainting){
            setRunningInpainting(true);
            var output = {
              text_prompts: text_prompts,
              negative_prompts: negative_prompts,
              batch_name: batch_name,
              steps: steps,
              aspect_ratio: aspect_ratio,
              width: width,
              height: height,
              seed:seed,
              use_random_seed:use_random_seed,
              n_samples: 1,
              n_iter: n_iter,
              cfg_scale: cfg_scale,
              sampler: sampler,
              init_image: init_image,
              strength: strength, 
              reverse_mask: reverse_mask,
              run_type: "inpainting"
            }
            window['updateSettings'](output).then((result) => {
              setImageB64("");
              setRunningInpainting(true);
          })
          }
        }

    return (
          <>
            {/* Center Portion */}
            <Box bg="gray.1000" p={4} width='100%' height="100%" rounded="md">
              <VStack spacing={4} align="center">
              <VStack>
                <ReactSketchCanvas
                  ref = {canvas}
                  style={styles}
                  strokeWidth={toolSize}
                  eraserWidth={toolSize}
                  strokeColor={hex}
                  canvasColor = {"white"}
                  // height={height+"px"}
                  // width={width+"px"} 
                  height="512px"
                  width="512px"
                  backgroundImage = {imageB64}
                />
              <HStack spacing="30px" ml="-40px">
                <IconButton background={ eraser ? "transparent": "gray.500"} onClick={(event) => {setEraser(false); canvas.current.eraseMode(false)}} aria-label='Pen' icon={<FaPen/>}></IconButton>
                <IconButton background={ eraser ? "gray.500" : "transparent"} onClick={(event) => {setEraser(true); canvas.current.eraseMode(true)}} aria-label='Eraser' icon={<BsEraserFill/>}></IconButton>
                <IconButton background={hex} onClick={(event) => setShowColorPicker(!showColorPicker)} aria-label='Choose a Color' icon={<IoColorPalette/>}></IconButton>
                <IconButton background="transparent" onClick={(event) => {if(toolSize>1) {setToolSize(toolSize-1)}}} aria-label='Decrease Size' icon={<FaMinus/>}></IconButton>
                <NumberInput min={1}            
                  id="toolSize"
                  name="toolSize"
                  variant="outline"
                  onChange={(v) => {setToolSize(v)}}
                  value={toolSize}
                  size = "s"
                  keepWithinRange={true}
                  >
                    <NumberInputField id='toolSize' rounded="md" height="35px" width="50px" pl={"3"}/>
                </NumberInput>
                <IconButton background="transparent" onClick={(event) => setToolSize(toolSize+1)} aria-label='Increase Size' icon={<FaPlus/>}></IconButton>
                <IconButton background="transparent" onClick={(event) => canvas.current.undo()} aria-label='Clear Init Image' icon={<FaUndoAlt/>}></IconButton>
                <IconButton background="transparent" onClick={(event) => canvas.current.redo()} aria-label='Clear Init Image' icon={<FaRedoAlt/>}></IconButton>
                <IconButton background="transparent" onClick={(event) => canvas.current.clearCanvas()} aria-label='Clear Init Image' icon={<FaTrash/>}></IconButton>
                </HStack>  
                  {
                    showColorPicker ?
                    <Colorful
                    style={{ marginLeft: 20 }}
                    color={hex}
                    onChange={(color) => {
                      setHex(color.hex);
                    }}
                    disableAlpha={false}
                  />:<></>
                  }
                </VStack>                
                <Box width='65%'>                
                  <Prompt/>
                </Box>                
                <HStack>
                  <Tooltip shouldWrapChildren  placement='top' label="Reverse the mask to replace everything EXCEPT the painted region. " fontSize='md'>
                      <FaQuestionCircle />
                    </Tooltip>
                  <Checkbox
                      id="reverse_mask"
                      name="reverse_mask"
                      onChange={() => {setReverseMask(!reverse_mask)}}
                      isChecked={reverse_mask}
                      colorScheme="purple"
                    >
                    Reverse Mask
                    </Checkbox>
                    <Button onClick = {submitInpainting} colorScheme="purple" ml={2} width="100px"
                      >
                      Run
                    </Button>
                    <Checkbox
                      id="open_pictures"
                      name="open_pictures"
                      onChange={() => {setOpenPictures(!open_pictures)}}
                      isChecked={open_pictures}
                      colorScheme="purple"
                    >
                    Open On Finish
                    </Checkbox>
                </HStack>
              </VStack>
            </Box>
        </>
    )
}
export default Inpainting;
