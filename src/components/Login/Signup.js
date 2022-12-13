import {useState} from 'react';
import {
    Flex,
    Heading,
    Input,
    Button,
    IconButton,
    InputGroup,
    Stack,
    InputLeftElement,
    Box,
    Link,
    FormControl,
    InputRightElement,
    Image,
    HStack,
} from '@chakra-ui/react';
import Select from 'react-select'
import { FaUserAlt, FaLock, FaEye, FaEyeSlash} from 'react-icons/fa';
import {IoIosMail} from 'react-icons/io'
import Logo from '../../images/ArtroomLogo.png';

const SignUp = ({setSignUp}) => {
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    
    const [showPassword, setShowPassword] = useState(false);
    const handleShowClick = () => setShowPassword(!showPassword);
    
    const months = [
        {label: 'January',  value: '01'},
        {label: 'February',  value: '02'},
        {label: 'March',  value: '03'},
        {label: 'April',  value: '04'},
        {label: 'May',  value: '05'},
        {label: 'June',  value: '06'},
        {label: 'July',  value: '07'},
        {label: 'August',  value: '08'},
        {label: 'September',  value: '09'},
        {label: 'October',  value: '10'},
        {label: 'November',  value: '11'},
        {label: 'December',  value: '12'},
      ];
    
      let days = [];
      for (let i = 1; i < 32; i++) {
        let iString = String(i);
        if (i<10){
            iString = '0'+iString;
        }
        days = days.concat({value: i, label: iString});
      }
    
      let years = []
      for (let i = 2022; i > 1900; i--) {
        years = years.concat({value: i, label: String(i)});
      }

    function handleSignUp(){
        setSignUp(false);
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
                            children={<FaUserAlt color='gray.800' />}
                        />
                        <Input 
                            borderColor= '#00000020' 
                            color='gray.800' 
                            type='text' 
                            placeholder='Username'
                            _placeholder={{ color: 'gray.400'}}
                        />
                        </InputGroup>
                    </FormControl>
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
                    />
                            <InputRightElement width='4.5rem'>
                                <IconButton color='gray.800' variant='ghost' icon={showPassword ? <FaEye /> : <FaEyeSlash />} h='1.75rem' size='sm' onClick={handleShowClick}>
                                </IconButton>
                            </InputRightElement>
                        </InputGroup>
                    </FormControl>
                    <FormControl>
                        <Flex justifyContent={'space-between'}>
                            <div style={{flex: 3, padding: 5}}> 
                                <Select
                                    options = {months}
                                    placeholder='Month'
                                    _placeholder={{ color: 'gray.400'}}
                                    styles={{
                                        option: (styles, state) => {
                                        const { isFocused, isSelected } = state;
                                        return {
                                            ...styles,
                                            color: isSelected ? 'white' : isFocused ? '#0000FF' : 'black',
                                            backgroundColor: isSelected ? '#0000FF' : isFocused ? '#E6E6FA' : 'white',
                                        };
                                        },
                                    }}
                                >
                                </Select>
                            </div>
                            <div style={{flex: 2, padding: 5}}> 
                                <Select
                                    options = {days}
                                    placeholder='Day'
                                    _placeholder={{ color: 'gray.400'}}
                                    styles={{
                                        option: (styles, state) => {
                                        const { isFocused, isSelected } = state;
                                        return {
                                            ...styles,
                                            color: isSelected ? 'white' : isFocused ? '#0000FF' : 'black',
                                            backgroundColor: isSelected ? '#0000FF' : isFocused ? '#E6E6FA' : 'white',
                                        };
                                        },
                                    }}
                                ></Select>
                            </div>          
                            <div style={{flex: 2, padding: 5}}> 
                                <Select
                                    options = {years}
                                    placeholder='Year'
                                    _placeholder={{ color: 'gray.400'}}
                                    styles={{
                                        option: (styles, state) => {
                                        const { isFocused, isSelected } = state;
                                        return {
                                            ...styles,
                                            color: isSelected ? 'white' : isFocused ? '#0000FF' : 'black',
                                            backgroundColor: isSelected ? '#0000FF' : isFocused ? '#E6E6FA' : 'white',
                                        };
                                        },
                                    }}
                                ></Select>
                            </div>
                        </Flex>
                    </FormControl>
                    <Button
                        borderRadius={10}
                        type='submit'
                        variant='solid'
                        colorScheme='blue'
                        width='full'
                        onClick={()=>{handleSignUp()}}
                    >
                        Sign Up
                    </Button>`
                </Stack>
            </Box>
        </Stack>
        <Box>
            Already have an account?{' '}
            <Link onClick={()=>{setSignUp(false)}} color='teal.500' href='#'>
                Login
            </Link>
        </Box>
        </Flex>

    )
}

export default SignUp;
