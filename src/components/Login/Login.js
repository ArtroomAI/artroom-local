import {useState} from 'react';
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
} from '@chakra-ui/react';
import {FaLock, FaEye, FaEyeSlash} from 'react-icons/fa';
import {IoIosMail} from 'react-icons/io';
import Logo from '../../images/ArtroomLogo.png';

const Login = ({setLoggedIn, setSignUp}) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [rememberMe, setRememberMe] = useState(false);

    const [showPassword, setShowPassword] = useState(false);

    const handleShowClick = () => setShowPassword(!showPassword);
    function handleLogin(){
        console.log(email);
        console.log(password);
        setLoggedIn(true)
    }
    
    return(
        <Flex
            flexDirection='column'
            width='100wh'
            height='60vh'
            justifyContent='center'
            alignItems='center'
            >
        <Stack
            flexDir='column'
            mb='2'
            justifyContent='center'
            alignItems='center'
            >
                <Image h='50px' src={Logo} />
                <Heading color='blue.600'>Welcome</Heading>
                <Box minW={{ base: '90%', md: '468px' }}>
                <Stack
                    spacing={4}
                    p='1rem'
                    backgroundColor='whiteAlpha.900'
                    boxShadow='md'
                >
                <FormControl>
                    <InputGroup>
                    <InputLeftElement
                        pointerEvents='none'
                        color='gray.800'
                        children={<IoIosMail color='gray.800' />}
                    />
                    <Input 
                        borderColor= '#00000020' 
                        color='gray.800' 
                        type='email' 
                        placeholder='Email Address'
                        _placeholder={{ color: 'gray.400'}}
                        value = {email}
                        onChange = {(event)=>setEmail(event.target.value)}
                    />
                    </InputGroup>
                </FormControl>
                <FormControl>
                    <InputGroup>
                        <InputLeftElement
                            pointerEvents='none'
                            color='gray.800'
                            children={<FaLock color='gray.800' />}
                        />
                        <Input
                            color='gray.800'
                            type={showPassword ? 'text' : 'password'}
                            placeholder='Password'
                            borderColor= '#00000020'
                            _placeholder={{ color: 'gray.400'}}
                            value = {password}
                            onChange = {(event)=>setPassword(event.target.value)}
                        />
                                <InputRightElement width='4.5rem'>
                                    <IconButton color='gray.800' variant='ghost' icon={showPassword ? <FaEye /> : <FaEyeSlash />} h='1.75rem' size='sm' onClick={handleShowClick}>
                                    </IconButton>
                                </InputRightElement>
                            </InputGroup>
                            <FormHelperText textAlign='right'>
                                <Link color='gray.800'>Forgot Password?</Link>
                            </FormHelperText>
                            </FormControl>
                            <Button
                                borderRadius={10}
                                type='submit'
                                variant='solid'
                                colorScheme='blue'
                                width='full'
                                onClick={handleLogin}
                            >
                                Login
                            </Button>
                        </Stack>
                </Box>
            </Stack>
            <Box>
                New to us?{' '}
                <Link onClick={()=>{setSignUp(true)}} color='teal.500' href='#'>
                    Sign Up
                </Link>
            </Box>
        </Flex>

    )
}

export default Login;
