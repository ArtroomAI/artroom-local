import { useState } from 'react';
import {
    Flex,
    Heading,
    Input,
    IconButton,
    Button,
    InputGroup,
    Stack,
    InputLeftElement,
    Box,
    Link,
    FormControl,
    FormHelperText,
    InputRightElement,
    Image
} from '@chakra-ui/react';
import { FaLock, FaEye, FaEyeSlash } from 'react-icons/fa';
import { IoIosMail } from 'react-icons/io';
import Logo from '../../images/ArtroomLogo.png';
import axios from 'axios';

const ForgotPasswordCode = ({ setLoggedIn, setState }) => {
    const [email, setEmail] = useState('');
    const [verificationCode, setVerificationCode] = useState('');

    function handleForgotPassword () {
        axios.post(
            'http://localhost:8000/login',
            qs.stringify({
                username: email,
                password,
                grant_type: '',
                scope: '',
                client_id: '',
                client_secret: ''
            }),
            {
                headers: { 'Content-Type': 'application/x-www-form-urlencoded',
                    'accept': 'application/json' }
            }
        ).then((result) => {
            console.log(result);
            // setState('ForgotPasswordCode');
        });
    }

    return (
        <Flex
            alignItems="center"
            flexDirection="column"
            height="60vh"
            justifyContent="center"
            width="100wh"
        >
            <Stack
                alignItems="center"
                flexDir="column"
                justifyContent="center"
                mb="2"
            >
                <Image
                    h="50px"
                    src={Logo} />

                <Heading color="blue.600">
                    Welcome
                </Heading>

                <Box minW={{ base: '90%',
                    md: '468px' }}>
                    <Stack
                        backgroundColor="whiteAlpha.900"
                        boxShadow="md"
                        p="1rem"
                        spacing={4}
                    >
                        <FormControl>
                            <InputGroup>
                                <InputLeftElement
                                    children={<IoIosMail color="gray.800" />}
                                    color="gray.800"
                                    pointerEvents="none"
                                />

                                <Input
                                    _placeholder={{ color: 'gray.400' }}
                                    borderColor="#00000020"
                                    color="gray.800"
                                    onChange={(event) => setEmail(event.target.value)}
                                    placeholder="Email Address"
                                    type="email"
                                    value={email}
                                />
                            </InputGroup>
                        </FormControl>

                        <Button
                            borderRadius={10}
                            colorScheme="blue"
                            onClick={handleForgotPassword}
                            type="submit"
                            variant="solid"
                            width="full"
                        >
                            Reset Password
                        </Button>
                    </Stack>
                </Box>
            </Stack>

            <Box>
                New to us?
                {' '}

                <Link
                    color="teal.500"
                    href="#"
                    onClick={() => {
                        setState('SignUp');
                    }}>
                    Sign Up
                </Link>
            </Box>
        </Flex>

    );
};

export default ForgotPasswordCode;
