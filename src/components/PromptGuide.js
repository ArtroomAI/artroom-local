import { React } from 'react';
import { Flex, VStack } from '@chakra-ui/layout';
import {
    Box,
    Heading,
    Container,
    Text,
    Stack
} from '@chakra-ui/react';

function PromptGuide () {
    return (
        <VStack p={5}>
            <Flex
                bg="gray.1000"
                ml="10px"
                w="100%">
                <Container maxW="3xl">
                    <Stack
                        as={Box}
                        py={{ base: 20,
                            md: 36 }}
                        spacing={{ base: 8,
                            md: 14 }}
                        textAlign="center">
                        <Heading
                            fontSize={{ base: '2xl',
                                sm: '4xl',
                                md: '6xl' }}
                            fontWeight={600}
                            lineHeight="110%">
                            Basics of Prompt Engineering
                            {' '}

                            <br />

                            <Text
                                as="span"
                                fontSize={{ base: '2xl',
                                    sm: '4xl',
                                    md: '6xl' }}
                                fontWeight={300}>
                                A guide by
                                {' '}

                                <br />
                            </Text>

                            <Text
                                color="green.400"
                                href="https://twitter.com/dailystablediff">
                                Graverman
                            </Text>

                            <Text
                                color="green.400"
                                fontSize="medium">
                                https://twitter.com/dailystablediff
                            </Text>
                        </Heading>

                        <Text color="gray.500">
                            Today I propose a simple formula for beginners to use and create better generations with text to image AI. This was tested on stable diffusion but it should work on any model if it was trained on enough art data.
                            <br />
                            After reading this document and applying these simple steps, you’ll be able to generate better images with the same amount of effort.
                        </Text>

                        <Text
                            color="gray.500"
                            fontSize={{ base: '2xl',
                                sm: '4xl',
                                md: '6xl' }}
                            fontWeight={300}>
                            1. Raw prompt
                        </Text>

                        <Text>
                            Raw prompt is the simplest way of describing what you want to generate, for instance;
                            <br />

                            1. Panda
                            <br />

                            2. A warrior with a sword
                            <br />

                            3. Skeleton
                            <br />

                            This is the basic building block of any prompt. Most new people start by only using raw prompts, this is usually a mistake as the images you generate like this tend to get random and chaotic. Here are some examples that I generated with running the earlier prompts
                        </Text>

                        <Text color="gray.500">
                            As you can see, these images have random scenery and don’t look very aesthetically pleasing, I definitely wouldn’t consider them art. This brings me to my next point;
                            <br />
                        </Text>

                        <Text
                            color="gray.500"
                            fontSize={{ base: '2xl',
                                sm: '4xl',
                                md: '6xl' }}
                            fontWeight={300}>
                            2. Style
                        </Text>

                        <Text>
                            Style is a crucial part of the prompt. The AI, when missing a specified style, usually chooses the one it has seen the most in related images, for example, if I generated landscape, it would probably generate realistic or oil painting looking images. Having a well chosen style + raw prompt is sometimes enough, as the style influences the image the most right after the raw prompt.
                            <br />

                            The most commonly used styles include:
                            <br />

                            1. Realistic

                            <br />

                            2. Oil painting

                            <br />

                            3. Pencil drawing

                            <br />

                            4. Concept art

                            <br />


                            I’ll examine them one by one to give an overview on how you might use these styles.
                            <br />

                            In the case of a realistic image, there are various ways of making it the style, most resulting in similar images. Here are some commonly used techniques of making the image realistic:
                            <br />

                            1. a photo of + raw prompt
                            <br />

                            2. a photograph of + raw prompt

                            <br />

                            3. raw prompt, hyperrealistic

                            <br />

                            4. raw prompt, realistic

                            <br />

                            You can of course combine these to get more and more realistic images.
                            <br />

                            To get oil painting you can just simply add “an oil painting of” to your prompt. This sometimes results in the image showing an oil painting in a frame, to fix this you can just re-run the prompt or use raw prompt + “oil painting”
                            <br />

                            To make a pencil drawing just simply add “a pencil drawing of” to your raw prompt or make your prompt raw prompt + “pencil drawing”.
                            <br />

                            The same applies to landscape art.
                            <br />
                        </Text>

                        <Text
                            color="gray.500"
                            fontSize={{ base: '2xl',
                                sm: '4xl',
                                md: '6xl' }}
                            fontWeight={300}>
                            3. Artist
                        </Text>

                        <Text>
                            <br />

                            To make your style more specific, or the image more coherent, you can use artists’ names in your prompt. For instance, if you want a very abstract image, you can add “made by Pablo Picasso” or just simply, “Picasso”.
                            <br />

                            Below are lists of artists in different styles that you can use, but I always encourage you to search for different artists as it is a cool way of discovering new art.
                            <br />

                            Portrait
                            <br />

                            1. John Singer Sargent
                            <br />

                            2. Edgar Degas
                            <br />

                            3. Paul Cézanne
                            <br />

                            4. Jan van Eyck
                            <br />


                            Oil painting
                            <br />

                            1. Leonardo DaVinci
                            <br />

                            2. Vincent Van Gogh

                            <br />

                            3. Johannes Vermeer

                            <br />

                            4. Rembrandt

                            <br />

                            Pencil/Pen drawing
                            <br />

                            1. Albrecht Dürer
                            <br />

                            2. Leonardo da Vinci

                            <br />

                            3. Michelangelo

                            <br />

                            4. Jean-Auguste-Dominique Ingres

                            <br />

                            Landscape art
                            <br />

                            1. Thomas Moran
                            <br />

                            2. Claude Monet

                            <br />

                            3. Alfred Bierstadt

                            <br />

                            4. Frederic Edwin Church

                            <br />

                            Mixing the artists is highly encouraged, as it can lead to interesting-looking art.
                            <br />
                        </Text>

                        <Text
                            color="gray.500"
                            fontSize={{ base: '2xl',
                                sm: '4xl',
                                md: '6xl' }}
                            fontWeight={300}>
                            4. Finishing touches
                        </Text>

                        <Text>
                            This is the part that some people take to extremes, leading to longer prompts than this article. Finishing touches are the final things that you add to your prompt to make it look like you want. For instance, if you want to make your image more artistic, add “trending on artstation”. If you want to add more realistic lighting add “Unreal Engine.” You can add anything you want, but here are some examples:
                            <br />

                            Highly detailed, surrealism, trending on art station, triadic color scheme, smooth, sharp focus, matte, elegant, the most beautiful image ever seen, illustration, digital paint, dark, gloomy, octane render, 8k, 4k, washed colors, sharp, dramatic lighting, beautiful, post processing, picture of the day, ambient lighting, epic composition
                            <br />

                        </Text>

                        <Text
                            color="gray.500"
                            fontSize={{ base: '2xl',
                                sm: '4xl',
                                md: '6xl' }}
                            fontWeight={300}>
                            5. Conclusion
                        </Text>

                        <Text>
                            Prompt engineering allows you to have better control of what the image will look like. It (if done right) improves the image quality by a lot in every aspect. If you enjoyed this “article”, well, I’m glad I didn’t waste my time. If you see any ways that I can improve this, definitely let me know on discord (Graverman#0804)
                            <br />
                            From the DreamStudio team: 'Thanks Graverman!'
                        </Text>
                    </Stack>
                </Container>
            </Flex>
        </VStack>
    );
}

export default PromptGuide;
