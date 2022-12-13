import {forwardRef} from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import ResizeTextarea from 'react-textarea-autosize';
import DragDropFile from './DragDropFile/DragDropFile';
import {
    Button,
    Box,
    Flex,
    FormControl,
    FormLabel,
    Textarea,
    HStack,
    Tooltip,
    VStack,
    Spacer  
  } from '@chakra-ui/react';
  import {
    FaQuestionCircle,
  } from 'react-icons/fa'

export const AutoResizeTextarea = forwardRef((props, ref) => {
  return (
    <Textarea
      spellCheck="false"
      minH="unset"
      overflow="hidden"
      w="100%"
      resize="none"
      ref={ref}
      minRows={1}
      as={ResizeTextarea}
      style={{       
        borderWidth: '1px',
        borderRadius: '1px',
        borderStyle: 'solid',
        borderColor: '#FFFFFF20'}}
      {...props}
    />
  );
});

function Prompt({setFocused}){
    const [text_prompts, setTextPrompts] = useRecoilState(atom.textPromptsState);
    const [negative_prompts, setNegativePrompts] = useRecoilState(atom.negativePromptsState);
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);

return(
    <> 
        <HStack width = '100%'>
          <VStack w = '80%'>
          <FormControl>
                <HStack w = '80%'>
                  <Tooltip shouldWrapChildren placement='top' label='Type here what you want the AI to generate in the image' fontSize='md'>
                      <FaQuestionCircle color='#777'/>
                </Tooltip>
                <FormLabel htmlFor='text_prompts'>Prompt</FormLabel>
                <Spacer/>
              {/* <Button h='25px' w = '100px' className='enhance-prompt' onClick={(event) => setPrompts(Use Prompt Enhance)}>
                Enhance
              </Button> */}
                </HStack>
                <Flex w='80%' className='text-prompts'>
                    <AutoResizeTextarea
                        id='text_prompts'
                        name='text_prompts'
                        variant='outline'
                        onChange = {(event) => setTextPrompts(event.target.value)}
                        value = {text_prompts}
                        onFocus={() => setFocused(true)}
                        onBlur={() => setFocused(false)}
                        />
                </Flex>
          </FormControl>
          <FormControl>
            <HStack w = '80%'>
            <Tooltip shouldWrapChildren  placement='top' label="Type what you DON'T want the AI to generate in the image" fontSize='md'>
                <FaQuestionCircle color='#777'/>
              </Tooltip>
            <FormLabel htmlFor='negative_prompts'>Negative Prompt</FormLabel>
            <Spacer/>
            <Button h='25px' w = '150px' className='defualt-negative-prompt' onClick={(event) => setNegativePrompts('lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry')}>
              Default Negative
            </Button>
          </HStack>
          <Spacer/>
          <Flex w='80%' className='negative-prompts'>
              <AutoResizeTextarea
                  id='negative_prompts'
                  name='negative_prompts'
                  variant='outline'
                  onChange = {(event) => setNegativePrompts(event.target.value)}
                  value = {negative_prompts}
                  onFocus={() => setFocused(true)}
                  onBlur={() => setFocused(false)}
                  />
          </Flex>
          </FormControl>
          </VStack>
          <VStack className='starting-image'>
            <HStack>
              <Tooltip shouldWrapChildren  placement='top' label='Upload an image to use as the starting point instead of just random noise' fontSize='md'>
                  <FaQuestionCircle color='#777'/>
              </Tooltip>
              <FormLabel htmlFor='init_image'>Starting Image</FormLabel>
            </HStack>
            <Box paddingBottom={30}>
              <DragDropFile handleFile={setInitImage}></DragDropFile>
            </Box>
          </VStack>
          </HStack>
        </>
    )
}

export default Prompt;
