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
    FormHelperText,
    Text,
    Image,
} from '@chakra-ui/react';
import Select from 'react-select'
import { FaUserAlt, FaLock, FaEye, FaEyeSlash} from 'react-icons/fa';
import {IoIosMail} from 'react-icons/io'
import Logo from '../../images/ArtroomLogo.png';
import validator from 'validator'

const SignUp = ({setSignUp}) => {
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [month, setMonth] = useState({});
    const [day, setDay] = useState({});
    const [year, setYear] = useState({});

    const [showPassword, setShowPassword] = useState(false);
    const handleShowClick = () => setShowPassword(!showPassword);
    
    const [usernameUnique, setUsernameUnique] = useState(true);
    const [emailValid, setEmailValid] = useState(true);
    const [emailUnique, setEmailUnique] = useState(true);
    const [passwordValid, setPasswordValid] = useState(true);
    const [above13, setAbove13] = useState(true);

    const checkValidEmail = () => {
        let isEmailValid = validator.isEmail(email) && email.length <= 100;
        setEmailValid(isEmailValid);
        return isEmailValid;    
    }

    const checkValidPassword = () => {
        let isPasswordValid = validator.isStrongPassword(password, {
                                    minLength: 8, minNumbers: 1, minSymbols: 1}) && password.length <= 100;
        setPasswordValid(isPasswordValid);
        return isPasswordValid;
    }

    const checkUniqueUsername = () => {
        return true;
        let isUsernameUnique = validator.isStrongPassword(username, {
                                    minLength: 8, minNumbers: 1, minSymbols: 1}) && username.length <= 100;
        setUsernameUnique(isUsernameUnique);
        return isUsernameUnique;
    }

    const checkUniqueEmail = () => {
        return true;
        let isEmailUnique = validator.isStrongPassword(email, {
                                    minLength: 8, minNumbers: 1, minSymbols: 1}) && email.length <= 100;
        setEmailUnique(isEmailUnique);
        return isEmailUnique;
    }

    const checkAbove13 = () => {
        const birthday = new Date(parseInt(year.value), parseInt(month.value) - 1, parseInt(day.value));

        // Get the current date
        const now = new Date();
        console.log(birthday, now)
        let age = now.getFullYear() - birthday.getFullYear();
        if (now.getMonth() < birthday.getMonth() || (now.getMonth() === birthday.getMonth() && now.getDate() < birthday.getDate())) {
            age -= 1;
        }

        setAbove13(age >= 13);
        return age >= 13;
    }

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
        var today = new Date();
        //Separate so they can all activate individually
        let isValidEmail = checkValidEmail();
        let isValidPassword = checkValidPassword();
        let isUniqueEmail = checkUniqueEmail();
        let isUniqueUsername = checkUniqueUsername();
        let isAbove13 = checkAbove13();

        if (isValidEmail && isValidPassword && isUniqueEmail && isUniqueUsername && isAbove13){
            console.log(username)
            console.log(email)
            console.log(password)
            console.log(month)
            console.log(day)
            console.log(year)
            setSignUp(false);
        }
        else{
            if (!checkValidEmail){
                console.log('Invalid Email');
                console.log(email);
            }
            if (!checkValidPassword){
                console.log('Invalid Password');
                console.log(password);
            }
        }
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
                        {!usernameUnique &&
                            <FormHelperText textAlign='left'>
                                <Text color='red.300'>Sorry, this username is taken.</Text>
                            </FormHelperText>}
                        <InputGroup>
                        <InputLeftElement
                            pointerEvents='none'
                            color='gray.800'
                            children={<FaUserAlt color='gray.800' />}
                        />
                        <Input 
                            borderColor = {usernameUnique ? '#00000020' : '#FF0000'} 
                            color='gray.800' 
                            type='text' 
                            placeholder='Username'
                            _placeholder={{ color: 'gray.400'}}
                            value = {username}
                            onChange = {(event)=>setUsername(event.target.value)}
                        />
                        </InputGroup>
                    </FormControl>
                    <FormControl>
                        {!emailValid &&
                        <FormHelperText textAlign='left'>
                            <Text color='red.300'>Please enter a valid email</Text>
                        </FormHelperText>}
                        {!emailUnique &&
                        <FormHelperText textAlign='left'>
                            <Text color='red.300'>Sorry, this email has been taken.</Text>
                        </FormHelperText>}
                        <InputGroup>
                        <InputLeftElement
                            pointerEvents='none'
                            color='gray.800'
                            children={<IoIosMail color='gray.800' />}
                        />
                        <Input 
                            borderColor = {emailValid && emailUnique ? '#00000020' : '#FF0000'} 
                            color='gray.800' 
                            type='email' 
                            placeholder='Email Address'
                            _placeholder={{ color: 'gray.400'}}
                            value = {email}
                            onChange = {(event)=>{
                                setEmailValid(true);
                                setEmail(event.target.value)}
                            }
                        />
                        </InputGroup>
                    </FormControl>
                    <FormControl>
                        {!passwordValid &&
                            <FormHelperText textAlign='left'>
                                <Text color='red.300'>Please enter a valid password. </Text>
                                <Text color='red.300'>
                                    Must at least 8 characters and include a number, and a symbol
                                </Text>
                            </FormHelperText>}
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
                        borderColor = {passwordValid ? '#00000020' : '#FF0000'} 
                        _placeholder={{ color: 'gray.400'}}
                        value = {password}
                        onChange = {(event)=>{
                            setPasswordValid(true);
                            setPassword(event.target.value);
                        }}
                            />
                            <InputRightElement width='4.5rem'>
                                <IconButton color='gray.800' variant='ghost' icon={showPassword ? <FaEye /> : <FaEyeSlash />} h='1.75rem' size='sm' onClick={handleShowClick}>
                                </IconButton>
                            </InputRightElement>
                        </InputGroup>
                    </FormControl>
                    <FormControl>
                        {!above13 &&
                            <FormHelperText textAlign='left'>
                                <Text color='red.300'>You must be at least 13 years old to use Artroom </Text>
                            </FormHelperText>}
                        <Flex justifyContent={'space-between'}>
                            <div style={{flex: 3, padding: 5}}> 
                                <Select
                                    options = {months}
                                    placeholder='Month'
                                    _placeholder={{ color: 'gray.400'}}
                                    value = {month}
                                    onChange = {(event)=>{setMonth(event)}}
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
                                    value = {day}
                                    onChange = {(event)=>setDay(event)}
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
                                    value = {year}
                                    onChange = {(event)=>setYear(event)}
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
