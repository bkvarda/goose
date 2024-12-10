import subprocess, os, json
from pathlib import Path
from functools import cached_property
import simple_salesforce
import logging
from pydantic.dataclasses import dataclass
import os

from exchange import Message  # type: ignore

from goose.toolkit.base import Toolkit, tool

class Salesforce(Toolkit):
    def system(self) -> str:
        """Retrieve detailed configuration and procedural guidelines for Salesforce operations"""
        template_content = Message.load("prompts/salesforce.jinja").text
        return template_content
    
    @cached_property
    def client(self) -> simple_salesforce.Salesforce:
        instance_url = os.getenv("SFDC_INSTANCE_URL")
        if not instance_url:
            raise Exception("SFDC_INSTANCE_URL environment variable not set")
        
        token_info = self.token_info()
        
        return Salesforce(
                          instance_url=instance_url,
                          session_id=token_info["accessToken"],
                          username=token_info["username"],
                          client_id = token_info["clientId"],
                          version= token_info["apiVersion"]
                          )
    
    @cached_property
    def token_info(self):
        s = subprocess.run(["sf", "force:org:display", "--json"],capture_output=True)
        if s.returncode != 0:
            #err = s.stderr.decode()
            #raise Exception(f"Failed to get SFDC token info: {err}")
            return self._login()
        else:
            return json.loads(s.stdout.decode())["result"]
    
     
    def _login(self):
        s = subprocess.run(["sfdx", "org", "login", "web", "--set-default", "--instance-url", self.instance_url], capture_output=True)
        if s.returncode != 0:
            err = s.stderr.decode()
            raise Exception(f"Failed to login to SFDC: {err}")
        
    @tool
    def list_objects(self) -> str:
        return str(self.client.describe()["sobjects"])
    
    @tool
    def query(self, query) -> str:
        return str(self.client.query_all(query)["records"])


    

