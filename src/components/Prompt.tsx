import React from 'react';
import { forwardRef } from 'react';
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
    Spacer,
    TextareaProps
} from '@chakra-ui/react';
import {
    FaQuestionCircle
} from 'react-icons/fa';

export const AutoResizeTextarea = forwardRef<HTMLTextAreaElement, TextareaProps>((props, ref) => (
    <Textarea
        as={ResizeTextarea}
        minH="unset"
        minRows={1}
        overflow="hidden"
        ref={ref}
        resize="none"
        spellCheck="false"
        style={{ borderWidth: '1px',
            borderRadius: '1px',
            borderStyle: 'solid',
            borderColor: '#FFFFFF20' }}
        w="100%"
        {...props}
    />
));

function Prompt ({ setFocused }: { setFocused: React.Dispatch<React.SetStateAction<boolean>> }) {
    const [text_prompts, setTextPrompts] = useRecoilState(atom.textPromptsState);
    const [negative_prompts, setNegativePrompts] = useRecoilState(atom.negativePromptsState);

    return (
        <>
            <HStack width="100%">
                <VStack w="80%">
                    <FormControl>
                        <HStack w="80%">
                            <Tooltip
                                fontSize="md"
                                label="Type here what you want the AI to generate in the image"
                                placement="top"
                                shouldWrapChildren>
                                <FaQuestionCircle color="#777" />
                            </Tooltip>

                            <FormLabel htmlFor="text_prompts">
                                Prompt
                            </FormLabel>

                            <Spacer />

                            {/* <Button h='25px' w = '100px' className='enhance-prompt' onClick={(event) => setPrompts(Use Prompt Enhance)}>
                Enhance
              </Button> */}
                        </HStack>

                        <Flex
                            className="text-prompts"
                            w="80%">
                            <AutoResizeTextarea
                                id="text_prompts"
                                name="text_prompts"
                                onBlur={() => setFocused(false)}
                                onChange={(event) => setTextPrompts(event.target.value)}
                                onFocus={() => setFocused(true)}
                                value={text_prompts}
                                variant="outline"
                            />
                        </Flex>
                    </FormControl>

                    <FormControl>
                        <HStack w="80%">
                            <Tooltip
                                fontSize="md"
                                label="Type what you DON'T want the AI to generate in the image"
                                placement="top"
                                shouldWrapChildren>
                                <FaQuestionCircle color="#777" />
                            </Tooltip>

                            <FormLabel htmlFor="negative_prompts">
                                Negative Prompt
                            </FormLabel>

                            <Spacer />

                            <Button
                                className="defualt-negative-prompt"
                                h="25px"
                                onClick={(event) => setNegativePrompts('lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry')}
                                w="150px">
                                Default Negative
                            </Button>
                        </HStack>

                        <Spacer />

                        <Flex
                            className="negative-prompts"
                            w="80%">
                            <AutoResizeTextarea
                                id="negative_prompts"
                                name="negative_prompts"
                                onBlur={() => setFocused(false)}
                                onChange={(event) => setNegativePrompts(event.target.value)}
                                onFocus={() => setFocused(true)}
                                value={negative_prompts}
                                variant="outline"
                            />
                        </Flex>
                    </FormControl>
                </VStack>

                <VStack className="starting-image">
                    <HStack>
                        <Tooltip
                            fontSize="md"
                            label="Upload an image to use as the starting point instead of just random noise"
                            placement="top"
                            shouldWrapChildren>
                            <FaQuestionCircle color="#777" />
                        </Tooltip>

                        <FormLabel htmlFor="init_image">
                            Starting Image
                        </FormLabel>
                    </HStack>

                    <Box paddingBottom={30}>
                        <DragDropFile />
                    </Box>
                </VStack>
            </HStack>
        </>
    );
}

export default Prompt;
