import React from 'react';
import { forwardRef } from 'react';
import { useRecoilState } from 'recoil';
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
import { DEFAULT_NEGATIVE_PROMPT, negativePromptsState, textPromptsState } from '../SettingsManager';

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
    const [textPrompts, setTextPrompts] = useRecoilState(textPromptsState);
    const [negativePrompts, setNegativePrompts] = useRecoilState(negativePromptsState);

    return (
        <HStack alignItems="flex-start" justifyContent="flex-start" width="100%">
        <VStack alignContent="start" w="100%">
            <FormControl>
            <HStack w="100%">
                <Tooltip
                fontSize="md"
                label="Type here what you want the AI to generate in the image"
                placement="top"
                shouldWrapChildren
                >
                <FaQuestionCircle color="#777" />
                </Tooltip>

                <FormLabel htmlFor="text_prompts">Prompt</FormLabel>

                <Spacer />
            </HStack>

            <Flex className="text-prompts" w="100%">
                <AutoResizeTextarea
                id="text_prompts"
                name="text_prompts"
                onBlur={() => setFocused(false)}
                onChange={(event) => setTextPrompts(event.target.value)}
                onFocus={() => setFocused(true)}
                value={textPrompts}
                variant="outline"
                />
            </Flex>
            </FormControl>

            <FormControl>
            <HStack w="100%">
                <Tooltip
                fontSize="md"
                label="Type what you DON'T want the AI to generate in the image"
                placement="top"
                shouldWrapChildren
                >
                <FaQuestionCircle color="#777" />
                </Tooltip>

                <FormLabel htmlFor="negative_prompts">Negative Prompt</FormLabel>

                <Spacer />

                <Button
                    className="defualt-negative-prompt"
                    variant="outline"
                    h="25px"
                    onClick={() =>
                        setNegativePrompts(DEFAULT_NEGATIVE_PROMPT)
                    }
                    w="150px"
                >
                Default Negative
                </Button>
            </HStack>

            <Spacer />

            <Flex className="negative-prompts" w="100%">
                <AutoResizeTextarea
                id="negative_prompts"
                name="negative_prompts"
                onBlur={() => setFocused(false)}
                onChange={(event) => setNegativePrompts(event.target.value)}
                onFocus={() => setFocused(true)}
                value={negativePrompts}
                variant="outline"
                />
            </Flex>
            </FormControl>
        </VStack>
        <Box pt="20px" pl="30">
            <DragDropFile />
        </Box>
        </HStack>
    );
}

export default Prompt;
