import React, { useState } from 'react';
import axios from 'axios';
import {
    Flex,
    Heading,
    Input,
    IconButton,
    Button,
    InputGroup,
    Stack,
    InputLeftElement,
    InputRightElement,
    Box,
    Link,
    Text,
    FormControl,
    FormHelperText,
    Image,
    createStandaloneToast
} from '@chakra-ui/react';
import { FaLock, FaEyeSlash, FaEye } from 'react-icons/fa';
import Logo from '../../images/ArtroomLogo.png';
import validator from 'validator';

const ResetPassword = ({ setState, pwdResetJwt }: { setState: React.Dispatch<React.SetStateAction<string>>, pwdResetJwt: string }) => {
    const SERVER_URL = process.env.REACT_APP_SERVER_URL;
    const { ToastContainer, toast } = createStandaloneToast();
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [passwordRepeat, setPasswordRepeat] = useState('');
    const [showPasswordRepeat, setShowPasswordRepeat] = useState(false); 
    const [passwordValid, setPasswordValid] = useState(true);

    const [samePassword, setSamePassword] = useState(true);

    const checkValidPassword = () => {
        const isPasswordValid = validator.isStrongPassword(
            password,
            { 
                minLength: 8,
                minNumbers: 1,
                minSymbols: 1,
                minUppercase: 0 
            }
        ) && password.length <= 100;
        setPasswordValid(isPasswordValid);
        return isPasswordValid;
    };

    function handleResetPassword () {
        const isValidPassword = checkValidPassword();
        if (password !== passwordRepeat){
            setSamePassword(false);
        }
        if (isValidPassword && password === passwordRepeat){
            let body = {
                token: pwdResetJwt,
                new_password: password
              }
            axios.post(
                `${SERVER_URL}/forgot_password_reset`,
                body,
                {   
                    headers: {
                        'Content-Type': 'application/json',
                        'accept': 'application/json'
                    }
                }
            ).then((result) => {
                console.log(result);
                toast({
                    title: 'Success',
                    description: "Password Successfully Reset",
                    status: 'success',
                    position: 'top',
                    duration: 3000,
                    isClosable: false,
                    containerStyle: {
                        pointerEvents: 'none'
                    }
                });
                setState('Login');
            }).catch(err => {
                console.log(err);
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
            });    
        }
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
                            {!samePassword &&
                            <FormHelperText textAlign="left">
                                <Text color="red.300">
                                    Passwords must match
                                </Text>
                            </FormHelperText>}
                            {!passwordValid &&
                            <FormHelperText textAlign="left">
                                <Text color="red.300">
                                    Please enter a valid password.
                                    {' '}
                                </Text>

                                <Text color="red.300">
                                    Must at least 8 characters and include a number, and a symbol
                                </Text>
                            </FormHelperText>}
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
                                    onChange={(event) => {
                                        setSamePassword(true);
                                        setPasswordValid(true);
                                        setPassword(event.target.value)
                                    }}
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
                                        onClick={() => setShowPassword(!showPassword)}
                                        size="sm"
                                        variant="ghost"
                                        aria-label={'show-password'} />
                                </InputRightElement>
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
                                    onChange={(event) => {
                                        setSamePassword(true);
                                        setPasswordValid(true);
                                        setPasswordRepeat(event.target.value)
                                    }}
                                    placeholder="Repeat Password"
                                    type={showPasswordRepeat
                                        ? 'text'
                                        : 'password'}
                                    value={passwordRepeat}
                                />
                                <InputRightElement width="4.5rem">
                                    <IconButton
                                        color="gray.800"
                                        h="1.75rem"
                                        icon={showPasswordRepeat
                                            ? <FaEye />
                                            : <FaEyeSlash />}
                                        onClick={() => setShowPasswordRepeat(!showPasswordRepeat)}
                                        size="sm"
                                        variant="ghost"
                                        aria-label={'show-password-repeat'} />
                                </InputRightElement>
                            </InputGroup>
                        </FormControl>
                        <Button
                            borderRadius={10}
                            colorScheme="blue"
                            onClick={handleResetPassword}
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

export default ResetPassword;
