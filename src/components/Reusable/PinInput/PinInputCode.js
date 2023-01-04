import React from 'react';
import {
  PinInput,
  PinInputField,
  FormControl,
  FormHelperText,
  Text,
  Link,
  HStack
} from "@chakra-ui/react";

export default function PinInputCode({
  verificationCode,
  setVerificationCode,
  numInputs,
  handleResendCode,
  ...props
}) {
  const fields = [<PinInputField key={0} borderColor="gray.800" color="gray.800" autoFocus />];
  for (let i = 1; i < numInputs; i++) {
    fields.push(<PinInputField key={i} borderColor="gray.800" color="gray.800" />);
  }

  const handleChange = (value) => {
    setVerificationCode(value);
  };

  return (
    <FormControl>
        <FormHelperText textAlign="left">
            <Text color="gray.800">
                Enter Your Code
            </Text>
        </FormHelperText>
        <HStack spacing={6} justifyContent="center">
          <PinInput
            placeholder={"•"}
            value={verificationCode}
            onChange={handleChange}
            alignContent="center" 
            spacing={4} 
            {...props}
          >
          {fields} 
          </PinInput>
        </HStack>
        <FormHelperText textAlign="right">
            <Link onClick={handleResendCode} color="teal">
                Resend code
            </Link>
        </FormHelperText>
    </FormControl>
  );
}
