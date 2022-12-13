import {React, useState, useEffect} from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../src/atoms/atoms';
import axios  from "axios";
import {
    Box,
    Button,
    Flex,
    VStack,
    createStandaloneToast 
  } from "@chakra-ui/react";
import ImageObj from '../src/components/ImageObj';
import Prompt from '../src/components/Prompt';

// import {
//   FaTrashAlt,
//   FaQuestionCircle,
//   FaChevronLeft,
//   FaChevronRight
// } from 'react-icons/fa'

  function Body() {
    const { ToastContainer, toast } = createStandaloneToast()
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);

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
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);
    const [strength, setStrength] = useRecoilState(atom.strengthState);

    const [queueRunning, setQueueRunning] = useRecoilState(atom.queueRunningState);
    const [image_save_path, setImageSavePath] = useRecoilState(atom.imageSavePathState);
    const [keep_warm, setKeepWarm] = useRecoilState(atom.keepWarmState);
    const [mainImage, setMainImage] = useRecoilState(atom.mainImageState);

    useEffect(() => {
      const interval = setInterval(() => 
        window['getImage']().then((result) => {
          setMainImage(result);
      }), 5000);
      return () => {
        clearInterval(interval);
      };
    }, [mainImage]);

    const submitMain = event => {
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
          reverse_mask: false,
          run_type: "regular"
        }
        if (!queueRunning || keep_warm){
          setQueueRunning(true);
          window['updateSettings'](output).then((result) => {
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
            axios.post('http://127.0.0.1:5000/add_to_queue',{
              text_prompts: text_prompts
            }, 
              {
                headers: {'Content-Type': 'application/json'
              }
            }).then((result) => { 
              console.log(result);
              toast({
                title: 'Completed!!',
                status: 'success',
                position: 'top',
                duration: 2000,
                isClosable: false,
                containerStyle: {
                  pointerEvents: 'none'
                },
              })   
              })
              .catch((error) => console.log(error));           
            // window['startSD']().then((result) => {
            //   //result === "message" from main.js
            //   let error_code = result["error_code"]
            //   if (error_code === "CUDA"){
            //     toast({
            //       title: 'CUDA OOM',
            //       description: "Ran out of VRAM loading/running model, try decreasing size of image or lowering generation speed", 
            //       status: 'error',
            //       position: 'top',
            //       duration: null,
            //       isClosable: true,
            //     })   
            //   }
            //   else if (error_code === "UNKNOWN"){
            //     toast({
            //       title: 'CUDA OOM',
            //       description: "Ran out of VRAM loading/running model, try decreasing size of image or lowering generation speed", 
            //       status: 'error',
            //       position: 'top',
            //       duration: null,
            //       isClosable: true,
            //     })   
            //   }
            //   setQueueRunning(false);
            //   });
          })
         
        }
      }

    return (
      <Flex transition="all .25s ease" ml={navSize === "large" ? "180px" : "100px"} align="center" justify="center" width='100%'> 
        <Box mt="20px" width='100%' align="center" >
            {/* Center Portion */}
              <VStack spacing={4} align="center">
                <Box className='image-box' width='75%' ratio={16 / 9}>
                  <ImageObj imagePath={mainImage["ImagePath"]} B64={mainImage["B64"]} active={true}></ImageObj>
                </Box>
                <Box width='65%'> 
                  <Prompt/>
                </Box>
                <Button className='run-button' onClick = {submitMain} ml={2} width="250px">{keep_warm ? "Add to Queue" : "Run"}</Button>
              </VStack>
          </Box>
        </Flex>
    )
}
export default Body;
