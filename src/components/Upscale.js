import {React, useState, useEffect} from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import axios  from 'axios';
import {
  createStandaloneToast,
  Box,
  Button,
  ButtonGroup,
  IconButton,
  Flex,
  FormControl,
  FormLabel,
  Input,
  VStack,
  HStack,
  Tooltip,
  NumberInputField, 
  NumberInput,
  Text,
  Stack,
  Select,
  Slider,
  SliderThumb,
  SliderFilledTrack,
  SliderTrack
} from '@chakra-ui/react';
import {
  FaQuestionCircle,
  FaTrashAlt
} from 'react-icons/fa';
function Upscale() {
  const { ToastContainer, toast } = createStandaloneToast()
  const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);
  const [upscale_images, setUpscaleImages] = useState('');
  const [upscale_dest, setUpscaleDest] = useState('');
  const [upscaler, setUpscaler] = useState('ESRGAN');
  const [upscale_factor, setUpscaleFactor] = useState(2);
  const [upscale_strength, setUpscaleStrength] = useState(0.5);

  const chooseUploadPath = event => {
    window['chooseImages']().then((result) => {
      setUpscaleImages(result);
    });
  }

  const chooseDestPath = event => {
    window['chooseUploadPath']().then((result) => {
      setUpscaleDest(result);
    });
  }

  const upscale = event => {
    toast({
      title: 'Recieved!',
      description: 'You\'ll get a notification 🔔 when your upscale is ready! (First time upscales make take longer while it downloads the models)',
      status: 'success',
      position: 'top',
      duration: 3000,
      isClosable: false,
      containerStyle: {
        pointerEvents: 'none'
      },
    })
    let output = {
        'upscale_images': upscale_images,
        'upscaler': upscaler,
        'upscale_factor': upscale_factor,
        'upscale_strength': upscale_strength,
        'upscale_dest': upscale_dest,
     }
    axios.post('http://127.0.0.1:5300/upscale',output, 
      {
        headers: {'Content-Type': 'application/json'}
      }).then((result) => { 
        if (result.data.status === 'Failure'){
          toast({
            title: 'Upscale Failed',
            description: result.data.status_message,
            status: 'error',
            position: 'top',
            duration: 5000,
            isClosable: false,
            containerStyle: {
              pointerEvents: 'none'
            },
          })     
      }
      else{
        toast({
          title: 'Upscale Completed',
          status: 'success',
          position: 'top',
          duration: 2000,
          isClosable: false,
          containerStyle: {
            pointerEvents: 'none'
          },
        })   
      }
    })
  }    


  return (
    <Flex transition='all .25s ease' ml={navSize === 'large' ? '100px' : '0px'} align='center' justify='center' width='100%'>
      <Box ml='30px' p={4} width='75%' height='90%' rounded='md'>
        <form>
          <VStack className='upscale' spacing={5} align='flex-start'>
            <FormControl className='upscale-images-input' width = 'full'>
                <HStack>
                  <Tooltip shouldWrapChildren mt='3' placement='right' 
                  label = 'You can select all images in folder with Ctrl+A, it will ignore folders in upscaling'
                    fontSize='md'>
                    <FaQuestionCircle color='#777'/>
                  </Tooltip>
                <FormLabel htmlFor='upscale_images'>Choose images to upscale</FormLabel>
                </HStack>
                <HStack>
                  <Input
                    id='upscale_images'
                    type='text'
                    name='upscale_images'
                    variant='outline'
                    onChange={(event) => setUpscaleImages(event.target.value)}
                    value = {upscale_images}
                  />
                  <ButtonGroup pl = '10px' variant='outline' isAttached>
                    <Button onClick = {chooseUploadPath}>Choose</Button>
                    <IconButton onClick = {(event) => setUpscaleImages('')} aria-label='Clear Init Image' icon={<FaTrashAlt/>}></IconButton>
                  </ButtonGroup>
                </HStack>
            </FormControl>

            <FormControl className='upscale-dest-input' width = 'full'>
                <FormLabel htmlFor='upscale_dest'>Choose upscale destination (leave blank to default to same directory as images)</FormLabel>
                <HStack>
                  <Input
                    id='upscale_dest'
                    type='text'
                    name='upscale_dest'
                    variant='outline'
                    onChange={(event) => setUpscaleDest(event.target.value)}
                    value = {upscale_dest}
                  />
                  <ButtonGroup pl = '10px' variant='outline' isAttached>
                    <Button onClick = {chooseDestPath}>Choose</Button>
                    <IconButton onClick = {(event) => setUpscaleDest('')} aria-label='Clear Init Image' icon={<FaTrashAlt/>}></IconButton>
                  </ButtonGroup>
                </HStack>
            </FormControl>

              <HStack >
                  <Tooltip shouldWrapChildren mt='3' placement='right' 
                  label = {<Stack>
                    <Text >GFPGAN fixes faces while upscaling </Text>
                    <Text >RealESRGAN is a general upscaler</Text>
                    <Text >RealESRGAN-anime is a better for anime</Text>
                </Stack>}
                    fontSize='md'>
                    <FaQuestionCircle color='#777'/>
                  </Tooltip>
                  <FormLabel htmlFor='upscaler'>Choose Upscaler</FormLabel>
                </HStack>
                <HStack>
                <FormControl className='upscaler-input'>
                  <Select                   
                    id='upscaler'
                    name='upscaler'
                    variant='outline'
                    onChange={(event) => setUpscaler(event.target.value)}
                    value={upscaler}
                    w = '300px'
                  >
                    <option style={{ backgroundColor: '#080B16' }} value='Choose Upscaler'>Choose Upscaler</option>
                    <option style={{ backgroundColor: '#080B16' }} value='GFPGANv1.3'>GFPGANv1.3</option>
                    <option style={{ backgroundColor: '#080B16' }} value='GFPGANv1.4'>GFPGANv1.4</option>
                    {/* <option value='CodeFormer'>CodeFormer</option> */}
                    <option style={{ backgroundColor: '#080B16' }} value='RestoreFormer'>RestoreFormer</option>
                    <option style={{ backgroundColor: '#080B16' }} value='RealESRGAN'>RealESRGAN</option>
                    <option style={{ backgroundColor: '#080B16' }} value='RealESRGAN-Anime'>RealESRGAN-Anime</option>
                  </Select>
                </FormControl>
                {/* {upscaler === 'CodeFormer' ?
                  <FormControl>
                    <HStack>
                    <Tooltip shouldWrapChildren  placement='top' label='CodeFormer Upscale Strength' fontSize='md'>
                        <FaQuestionCircle color='#777'/>
                        </Tooltip>
                        <FormLabel  htmlFor='upscale_strength'>Upscale Strength:</FormLabel> 
                    </HStack>
                    <Slider         
                        id='upscale_strength'
                        name='upscale_strength'
                        variant='outline'
                        defaultValue={0.5} min={0.01} max={1.0} step={0.01} colorScheme='teal'
                        onChange={(v) => {setUpscaleStrength(v)}}
                        >
                        <SliderTrack bg='red.100'>
                          <Box position='relative' right={10} />
                            <SliderFilledTrack bg='tomato' />
                          </SliderTrack>
                        <Tooltip
                          bg='teal.500'
                          color='white'
                          placement='right'
                          isOpen={true}
                          label={`${upscale_strength}`}
                          >
                        <SliderThumb />
                        </Tooltip>                
                    </Slider>
                    </FormControl>
                      : <></>
                    } */}
              </HStack>

            <FormControl className='upscale-factor-input'>
              <HStack>
                <Tooltip shouldWrapChildren  placement='top' 
                label= {<Stack>
                  <Text >Multiplies your Width and Height</Text>
                  <Text >(ex: a multiplier of 2 will make 512x512 into 1024x1024)</Text>
                </Stack>}
                  fontSize='md'>
                  <FaQuestionCircle color='#777'/>
                </Tooltip>
                <FormLabel htmlFor='upscale_factor'>Upscale Multiplier (whole numbers only)</FormLabel>
              </HStack>
              <NumberInput min={1} step={1}     
                id='upscale_factor'
                name='upscale_factor'
                variant='outline'
                onChange={(v) => {setUpscaleFactor(v)}}
                value={upscale_factor}
                w = '200px'
                >
                  <NumberInputField id='upscale_factor' />
              </NumberInput>
            </FormControl>  
            

            <Button className='upscale-button' onClick = {upscale}>
              Upscale
            </Button>
          </VStack>
        </form>
      </Box>
    </Flex>
  )
}

export default Upscale;
