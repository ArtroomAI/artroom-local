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
    Image,
    createStandaloneToast
} from '@chakra-ui/react';
import { FaLock, FaEye, FaEyeSlash } from 'react-icons/fa';
import { IoIosMail } from 'react-icons/io';
import Logo from '../../images/ArtroomLogo.png';
import axios from 'axios';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms'
import ProtectedReqManager from '../../helpers/ProtectedReqManager';

const Login = ({ setLoggedIn, setState}) => {    
    const ARTROOM_URL = process.env.REACT_APP_SERVER_URL;
    const { ToastContainer, toast } = createStandaloneToast();
    const qs = require('qs');

    const [email, setEmail] = useRecoilState(atom.emailState);

    const [password, setPassword] = useState('');

    const [showPassword, setShowPassword] = useState(false);

    const handleShowClick = () => setShowPassword(!showPassword);

    function handleResendCode(){
        axios.get(
            `${ARTROOM_URL}/resend_code`,
            {
                params : {
                    email,
                },
                headers: {
                    'Content-Type': 'application/json',
                    'accept': 'application/json'
                }
            }
        ).then((result) => {
            console.log(result);
            toast({
                title: 'Sent Verification Code',
                description: "Please check your email (it may have went to spam)",
                status: 'success',
                position: 'top',
                duration: 2000,
                isClosable: false,
                containerStyle: {
                    pointerEvents: 'none'
                }
            });
        });  
    }

    function handleLogin () {
        axios.post(
            `${ARTROOM_URL}/login`,
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
            
            ProtectedReqManager.set_access_token(result.data.access_token)
            ProtectedReqManager.set_refresh_token(result.data.refresh_token)
            setLoggedIn(true);
        }).catch(err => {
            console.log(err);
            if (err.response.data.detail === "user unverified"){
                handleResendCode();
                setState("EmailVerificationCode");
            } else {
                toast({
                    title: "Error",
                    description: err.response.data.detail,
                    status: 'error',
                    position: 'top',
                    duration: 2000,
                    isClosable: false,
                    containerStyle: {
                        pointerEvents: 'none'
                    }
                });
            }
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

                        <FormControl>
                            <InputGroup>
                                <InputLeftElement
                                    children={<FaLock color="gray.800" />}
                                    color="gray.800"
                                    pointerEvents="none"
                                />

                                <Input
                                    _placeholder={{ color: 'gray.400' }}
                                    borderColor="#00000020"
                                    color="gray.800"
                                    onChange={(event) => setPassword(event.target.value)}
                                    placeholder="Password"
                                    type={showPassword
                                        ? 'text'
                                        : 'password'}
                                    value={password}
                                />

                                <InputRightElement width="4.5rem">
                                    <IconButton
                                        color="gray.800"
                                        h="1.75rem"
                                        icon={showPassword
                                            ? <FaEye />
                                            : <FaEyeSlash />}
                                        onClick={handleShowClick}
                                        size="sm"
                                        variant="ghost" />
                                </InputRightElement>
                            </InputGroup>

                            <FormHelperText textAlign="right">
                                <Link onClick={() => {
                                        setState('ForgotPassword');
                                     }}               
                                     color="gray.800">
                                    Forgot Password?
                                </Link>
                            </FormHelperText>
                        </FormControl>

                        <Button
                            borderRadius={10}
                            colorScheme="blue"
                            onClick={handleLogin}
                            type="submit"
                            variant="solid"
                            width="full"
                        >
                            Login
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

export default Login;
