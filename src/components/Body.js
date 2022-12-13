import {React, useState, useEffect, useCallback, useReducer} from 'react';
import { useInterval } from './Reusable/useInterval/useInterval';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import axios  from 'axios';
import {
    Box,
    Button,
    Flex,
    VStack,
    Progress,
    SimpleGrid,
    Image,
    createStandaloneToast
  } from '@chakra-ui/react';
import ImageObj from './ImageObj';
import Prompt from './Prompt';

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
    const [n_iter, setNIter] = useRecoilState(atom.nIterState);
    const [sampler, setSampler] = useRecoilState(atom.samplerState);
    const [cfg_scale, setCFGScale] = useRecoilState(atom.CFGScaleState);
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);
    const [ckpt, setCkpt] = useRecoilState(atom.ckptState);
    const [image_save_path, setImageSavePath] = useRecoilState(atom.imageSavePathState);
    const [long_save_path, setLongSavePath] = useRecoilState(atom.longSavePathState);
    const [highres_fix, setHighresFix] = useRecoilState(atom.highresFixState);
    const [speed, setSpeed] = useRecoilState(atom.speedState);
    const [use_full_precision, setUseFullPrecision] = useRecoilState(atom.useFullPrecisionState);
    const [use_cpu, setUseCPU] = useRecoilState(atom.useCPUState);
    const [save_grid, setSaveGrid] = useRecoilState(atom.saveGridState);
    const [debug_mode, setDebugMode] = useRecoilState(atom.debugMode);
    const [ckpt_dir, setCkptDir] = useRecoilState(atom.ckptDirState);
    const [strength, setStrength] = useRecoilState(atom.strengthState);
    const [delay, setDelay] = useRecoilState(atom.delayState);

    const [mainImage, setMainImage] = useRecoilState(atom.mainImageState);
    const [latestImages, setLatestImages] = useRecoilState(atom.latestImageState);
    const [latestImagesID, setLatestImagesID] = useRecoilState(atom.latestImagesIDState);

    const [progress, setProgress] = useState(-1);
    const [stage, setStage] = useState('');
    const [running, setRunning] = useState(false);
    const [focused, setFocused] = useState(false);

    const mainImageIndex = { selectedIndex: 0 };
    const reducer = (state, action) => {
      switch (action.type) {
        case 'arrowLeft':
          console.log('Arrow Left');
          return {
            selectedIndex:
              state.selectedIndex !== 0 ? state.selectedIndex - 1 : latestImages.length - 1,
          };
        case 'arrowRight':
          console.log('Arrow Right')
          return {
            selectedIndex:
              state.selectedIndex !== latestImages.length - 1 ? state.selectedIndex + 1 : 0,
          };
        case 'select':
          console.log('Select')
          return { selectedIndex: action.payload };
        default:
          throw new Error();
      }
    };

    const [state, dispatch] = useReducer(reducer, mainImageIndex);

    const useKeyPress = (targetKey, useAltKey = false) => {
      const [keyPressed, setKeyPressed] = useState(false);

      useEffect(() => {
        const leftHandler = ({ key, altKey}) => {
          if (key === targetKey && altKey === useAltKey) {
            console.log(key);
            console.log(altKey);
            setKeyPressed(true);
          }
        };

        const rightHandler = ({ key, altKey}) => {
          if (key === targetKey && altKey === useAltKey) {
            setKeyPressed(false);
          }
        };

        window.addEventListener('keydown', leftHandler);
        window.addEventListener('keyup', rightHandler);

        return () => {
          window.removeEventListener('keydown', leftHandler);
          window.removeEventListener('keyup', rightHandler);
        };
      }, [targetKey]);

      return keyPressed;
    };

    const arrowRightPressed = useKeyPress('ArrowRight');
    const arrowLeftPressed = useKeyPress('ArrowLeft');
    const altRPressed = useKeyPress('r', true);

  useEffect(() => {
    if (arrowRightPressed && !focused) {
      dispatch({ type: 'arrowRight' });
    }
  }, [arrowRightPressed]);

  useEffect(() => {
    if (arrowLeftPressed && !focused) {
      dispatch({ type: 'arrowLeft' });
    } 
  }, [arrowLeftPressed]);

  useEffect(() => {
    if (altRPressed) {
      submitMain();
    }
  }, [altRPressed]);

  useEffect(()=>{
    setMainImage(latestImages[state.selectedIndex])
  },[state])

    useEffect(() => {
      const interval = setInterval(() =>
      axios.get('http://127.0.0.1:5300/get_progress',
      {headers: {'Content-Type': 'application/json'}}).then((result)=>{
        if (result.data.status === 'Success'){
          setProgress(result.data.content.percentage);
          setRunning(result.data.content.running);
          setStage(result.data.content.stage);
          if (result.data.content.status === 'Loading Model' && !toast.isActive('loading-model')) {
            toast({
              id: 'loading-model',
              title: 'Loading model...',
              status: 'info',
              position: 'bottom-right',
              duration: 30000,
              isClosable: false,
            })
          }
          if (!(result.data.content.status === 'Loading Model')){
            if (toast.isActive('loading-model')){
              toast.close('loading-model')
            }
          }
        }
        else{
          setProgress(-1);
          setStage('');
          setRunning(false);
          if (toast.isActive('loading-model')){
            toast.close('loading-model')
          }
        }
      }), 1500);
      return () => {
        clearInterval(interval);
      };
    }, []);


    useInterval(() => {
      axios.get('http://127.0.0.1:5300/get_images',
      {params: {'path': 'latest', 'id': latestImagesID},
      headers: {'Content-Type': 'application/json'}}
      ).then((result) => {
          let id = result.data.content.latest_images_id;
          // console.log(id);
          // console.log(latestImagesID);
          if (result.data.status === 'Success'){
            if (id !== latestImagesID){
              setLatestImagesID(id);
              setLatestImages(result.data.content.latest_images);
              setMainImage(result.data.content.latest_images[result.data.content.latest_images.length-1]);
            }
          }
        else if (result.data.status === 'Failure'){
            setMainImage('');
          }
  })
    }, 3000);

    const submitMain = event => {
      axios.post('http://127.0.0.1:5300/add_to_queue',{
        text_prompts: text_prompts,
        negative_prompts: negative_prompts,
        batch_name: batch_name,
        steps: steps,
        aspect_ratio: aspect_ratio,
        width: width,
        height: height,
        seed:seed,
        use_random_seed:use_random_seed,
        n_iter: n_iter,
        cfg_scale: cfg_scale,
        sampler: sampler,
        init_image: init_image,
        strength: strength,
        reverse_mask: false,
        ckpt: ckpt,
        image_save_path: image_save_path,
        long_save_path: long_save_path,
        highres_fix: highres_fix,
        speed: speed,
        use_full_precision: use_full_precision,
        use_cpu: use_cpu,
        save_grid: save_grid,
        debug_mode: debug_mode,
        ckpt_dir: ckpt_dir,
        mask: '',
        delay: delay,
      },
        {
          headers: {'Content-Type': 'application/json'
        }
      }).then((result) => {
        if (result.data.status === 'Success'){
          toast({
            title: 'Added to Queue!',
            status: 'success',
            position: 'top',
            duration: 2000,
            isClosable: false,
            containerStyle: {
              pointerEvents: 'none'
            },
          })
        }
        else {
          toast({
            title: 'Error',
            status: 'error',
            description: result.data.status_message,
            position: 'top',
            duration: 5000,
            isClosable: true,
            containerStyle: {
              pointerEvents: 'none'
            },
          })
        }}).catch((error) => console.log(error));
    }

    return (
      <Flex transition='all .25s ease' ml={navSize === 'large' ? '180px' : '100px'} width='100%'>
        <Box width='100%' align='center' >
            {/* Center Portion */}
              <VStack spacing={4}>
                <Box  className='image-box' width='75%' ratio={16 / 9}>
                  <ImageObj B64={mainImage} active={true}></ImageObj>
                  {
                    progress >= 0 ? <Progress align='left' hasStripe value={progress} /> : <></>
                  }
                </Box>
                <Box width='50%' overflowY="auto" maxHeight="120px">
                  <SimpleGrid minChildWidth='100px' spacing='10px'>
                    {latestImages?.map((image,index)=>{
                        return(
                          <Image
                            key = {index} h='5vh' fit='scale-left' src={image}
                            onClick={() => dispatch({ type: 'select', payload: index })}
                            />
                          )
                        })
                      }
                  </SimpleGrid>
                </Box>
                  <Button className='run-button' onClick = {submitMain} ml={2} width='250px'>{running ? 'Add to Queue' : 'Run'}</Button>
                  <Box width='80%'>
                    <Prompt setFocused={setFocused}/>
                  </Box>
              </VStack>
          </Box>
        </Flex>
    )
}
export default Body;
