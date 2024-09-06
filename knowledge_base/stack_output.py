import boto3
import json


def get_stack_outputs(stack_name: str, region_name: str = 'us-east-1'):
              
            client = boto3.client('cloudformation', region_name=region_name)
        

            try:
                response = client.describe_stacks(StackName=stack_name)
                stacks = response.get('Stacks', [])
                if not stacks:
                    raise ValueError(f"No stack found with name {stack_name}")

                stack = stacks[0]
                outputs = stack.get('Outputs', [])
               
                return {output['OutputKey']: output['OutputValue'] for output in outputs}

            except Exception as e:
                print(f"Error retrieving stack outputs: {e}")
                return {}

if __name__ == "__main__":
            # Replace with your stack name and region
     
            stack_name = "MmragStack"
            region_name = 'us-east-1'

            outputs = get_stack_outputs(stack_name, region_name)

            # Print the outputs or save them to a file
            print(json.dumps(outputs, indent=4))

            # Optionally, save to a file
            with open('stack_outputs.json', 'w') as f:
                json.dump(outputs, f, indent=4)



            

            
            