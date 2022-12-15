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

const Login = ({ setLoggedIn, setSignUp }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [rememberMe, setRememberMe] = useState(false);

    const [showPassword, setShowPassword] = useState(false);

    const handleShowClick = () => setShowPassword(!showPassword);
    function handleLogin () {
        console.log(email);
        console.log(password);
        setLoggedIn(true);
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
                                <Link color="gray.800">
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
                        setSignUp(true);
                    }}>
                    Sign Up
                </Link>
            </Box>
        </Flex>

    );
};

export default Login;
